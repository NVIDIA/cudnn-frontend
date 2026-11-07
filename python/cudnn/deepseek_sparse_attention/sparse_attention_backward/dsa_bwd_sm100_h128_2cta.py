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
from cutlass.cute.typing import BFloat16, Float32, Int32, Int64
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator, TmemAllocator
from ._rank_common_sm100 import _partial_cta_sync
from .dsa_bwd_sm100_2cta_deterministic import TwoCTADeterministicReduction, ordered_dkv_quad
from .dsa_bwd_sm100_2cta_rank_shard import (
    RANK_CODE_INDIRECT,
    RANK_CODE_STORE,
    RANK_CODE_VALUE,
    RANK_INTERLEAVE_MIN_PLANES,
    RANK_STEP_BIAS,
    RankShardReduction,
    rank_reduce_f32x4,
    rank_reduce_f32x4_stream,
    rank_store_f32x4,
    rank_store_f32x4_stream,
)

U64x4 = Tuple[cutlass.Uint64, cutlass.Uint64, cutlass.Uint64, cutlass.Uint64]
F32x8 = Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]
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
def _bf16x2_to_f32x2(word: cutlass.Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Widen one BF16x2 word into its low and high logical values exactly."""
    result_type = llvm.StructType.get_literal([T.f32(), T.f32()])
    values = llvm.inline_asm(
        result_type,
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xFFFF0000;\nmov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (Float32(llvm.extractvalue(T.f32(), values, [0], loc=loc, ip=ip)), Float32(llvm.extractvalue(T.f32(), values, [1], loc=loc, ip=ip)))


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
    return tuple((cutlass.Uint64(llvm.extractvalue(T.i64(), value, [index], loc=loc, ip=ip)) for index in range(4)))


def _extract_f32x16(value, *, loc=None, ip=None) -> F32x16:
    return tuple((Float32(llvm.extractvalue(T.f32(), value, [index], loc=loc, ip=ip)) for index in range(16)))


@dsl_user_op
def _prefetch_o_row_l2(source: cute.Pointer, *, loc=None, ip=None) -> None:
    """Pull one immutable O row chunk into L2 ahead of the reducer sweep."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(None, [source_i64], "prefetch.global.L2 [$0];", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _prefetch_bulk_l2(source: cute.Pointer, size_bytes: Int32, *, loc=None, ip=None) -> None:
    """Hint an asynchronous L2 prefetch of ``size_bytes`` (a multiple of 16) from a 16-byte aligned address."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [source_i64, Int32(size_bytes).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
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
def _load_o_bf16x16_read_once(source: cute.Pointer, *, loc=None, ip=None) -> U64x4:
    """Read-once O load for the deterministic O*dO statistics; pure because O is immutable during this kernel.

    Each O row is read exactly once, so it bypasses L1 and is the first L2 eviction candidate:
    the KV rows, dense-plane rows and Q/dO round operands that later tiles and waves re-read stay.
    """
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i64()] * 4)
    value = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.nc.L1::no_allocate.L2::evict_first.v4.u64 {$0, $1, $2, $3}, [$4];",
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
def _decode_bf16x16_to_f32(x0: cutlass.Uint64, x1: cutlass.Uint64, x2: cutlass.Uint64, x3: cutlass.Uint64, *, loc=None, ip=None) -> F32x16:
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
        (
            "{\n"
            "        .reg .b32 word<8>;\n"
            "        .reg .b16 half<16>;\n"
            "        mov.b64 {word0, word1}, $16;\n"
            "        mov.b64 {word2, word3}, $17;\n"
            "        mov.b64 {word4, word5}, $18;\n"
            "        mov.b64 {word6, word7}, $19;\n"
            "        mov.b32 {half0, half1}, word0;\n"
            "        mov.b32 {half2, half3}, word1;\n"
            "        mov.b32 {half4, half5}, word2;\n"
            "        mov.b32 {half6, half7}, word3;\n"
            "        mov.b32 {half8, half9}, word4;\n"
            "        mov.b32 {half10, half11}, word5;\n"
            "        mov.b32 {half12, half13}, word6;\n"
            "        mov.b32 {half14, half15}, word7;\n"
            "        cvt.f32.bf16 $0, half0;\n"
            "        cvt.f32.bf16 $1, half1;\n"
            "        cvt.f32.bf16 $2, half2;\n"
            "        cvt.f32.bf16 $3, half3;\n"
            "        cvt.f32.bf16 $4, half4;\n"
            "        cvt.f32.bf16 $5, half5;\n"
            "        cvt.f32.bf16 $6, half6;\n"
            "        cvt.f32.bf16 $7, half7;\n"
            "        cvt.f32.bf16 $8, half8;\n"
            "        cvt.f32.bf16 $9, half9;\n"
            "        cvt.f32.bf16 $10, half10;\n"
            "        cvt.f32.bf16 $11, half11;\n"
            "        cvt.f32.bf16 $12, half12;\n"
            "        cvt.f32.bf16 $13, half13;\n"
            "        cvt.f32.bf16 $14, half14;\n"
            "        cvt.f32.bf16 $15, half15;\n"
            "        }"
        ),
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
        (
            "{\n"
            "        .reg .f32 acc;\n"
            "        mov.f32 acc, 0f00000000;\n"
            "        fma.rn.f32 acc, $1, $17, acc;\n"
            "        fma.rn.f32 acc, $2, $18, acc;\n"
            "        fma.rn.f32 acc, $3, $19, acc;\n"
            "        fma.rn.f32 acc, $4, $20, acc;\n"
            "        fma.rn.f32 acc, $5, $21, acc;\n"
            "        fma.rn.f32 acc, $6, $22, acc;\n"
            "        fma.rn.f32 acc, $7, $23, acc;\n"
            "        fma.rn.f32 acc, $8, $24, acc;\n"
            "        fma.rn.f32 acc, $9, $25, acc;\n"
            "        fma.rn.f32 acc, $10, $26, acc;\n"
            "        fma.rn.f32 acc, $11, $27, acc;\n"
            "        fma.rn.f32 acc, $12, $28, acc;\n"
            "        fma.rn.f32 acc, $13, $29, acc;\n"
            "        fma.rn.f32 acc, $14, $30, acc;\n"
            "        fma.rn.f32 acc, $15, $31, acc;\n"
            "        fma.rn.f32 acc, $16, $32, acc;\n"
            "        mov.f32 $0, acc;\n"
            "        }"
        ),
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
    """Reduce within eight 4-lane row groups rather than one 32-lane tree."""
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
        (
            "{\n"
            "        .reg .u32 lane, bits, send_bits, recv_bits;\n"
            "        .reg .pred pred;\n"
            "        .reg .f32 p<8>, y<4>, z<2>, lhs, rhs, acc, recv;\n"
            "        mov.f32 p0, $1;\n"
            "        mov.f32 p1, $2;\n"
            "        mov.f32 p2, $3;\n"
            "        mov.f32 p3, $4;\n"
            "        mov.f32 p4, $5;\n"
            "        mov.f32 p5, $6;\n"
            "        mov.f32 p6, $7;\n"
            "        mov.f32 p7, $8;\n"
            "        mov.u32 lane, %laneid;\n"
            "\n"
            "        and.b32 bits, lane, 16;\n"
            "        setp.ne.u32 pred, bits, 0;\n"
            "        selp.f32 lhs, p4, p0, pred;\n"
            "        selp.f32 rhs, p0, p4, pred;\n"
            "        add.rn.f32 y0, lhs, rhs;\n"
            "        selp.f32 lhs, p5, p1, pred;\n"
            "        selp.f32 rhs, p1, p5, pred;\n"
            "        add.rn.f32 y1, lhs, rhs;\n"
            "        selp.f32 lhs, p6, p2, pred;\n"
            "        selp.f32 rhs, p2, p6, pred;\n"
            "        add.rn.f32 y2, lhs, rhs;\n"
            "        selp.f32 lhs, p7, p3, pred;\n"
            "        selp.f32 rhs, p3, p7, pred;\n"
            "        add.rn.f32 y3, lhs, rhs;\n"
            "\n"
            "        and.b32 bits, lane, 8;\n"
            "        setp.ne.u32 pred, bits, 0;\n"
            "        selp.f32 lhs, y2, y0, pred;\n"
            "        selp.f32 rhs, y0, y2, pred;\n"
            "        add.rn.f32 z0, lhs, rhs;\n"
            "        selp.f32 lhs, y3, y1, pred;\n"
            "        selp.f32 rhs, y1, y3, pred;\n"
            "        add.rn.f32 z1, lhs, rhs;\n"
            "\n"
            "        and.b32 bits, lane, 4;\n"
            "        setp.ne.u32 pred, bits, 0;\n"
            "        selp.f32 lhs, z1, z0, pred;\n"
            "        selp.f32 rhs, z0, z1, pred;\n"
            "        add.rn.f32 acc, lhs, rhs;\n"
            "\n"
            "        mov.b32 send_bits, acc;\n"
            "        shfl.sync.bfly.b32 recv_bits, send_bits, 2, 0x1f, 0xffffffff;\n"
            "        mov.b32 recv, recv_bits;\n"
            "        add.rn.f32 acc, acc, recv;\n"
            "        mov.b32 send_bits, acc;\n"
            "        shfl.sync.bfly.b32 recv_bits, send_bits, 1, 0x1f, 0xffffffff;\n"
            "        mov.b32 recv, recv_bits;\n"
            "        add.rn.f32 acc, acc, recv;\n"
            "        mov.f32 $0, acc;\n"
            "        }"
        ),
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
def _load_f32x8_cg(source: cute.Pointer, *, loc=None, ip=None) -> F32x8:
    """Load eight FP32 values through L2 using the ABI's 16-byte alignment."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 8)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$8];\nld.global.cg.v4.f32 {$4, $5, $6, $7}, [$8+16];",
        "=f,=f,=f,=f,=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple((Float32(llvm.extractvalue(T.f32(), values, [index], loc=loc, ip=ip)) for index in range(8)))


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
def _store_f32x4_zero(destination: cute.Pointer, *, loc=None, ip=None) -> None:
    """Store four positive-zero FP32 bit patterns with one 128-bit instruction."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    zero = Int32(0).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [destination_i64, zero],
        "st.global.v4.b32 [$0], {$1, $1, $1, $1};",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
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


@dsl_user_op
def _finite_residual_f32(value: Float32, high: Float32, *, loc=None, ip=None) -> Float32:
    """Return value-high for finite high, otherwise zero, without staged branches."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip), Float32(high).ir_value(loc=loc, ip=ip)],
            "{\n        .reg .pred finite;\n        .reg .f32 low;\n        testp.finite.f32 finite, $2;\n        sub.rn.f32 low, $1, $2;\n        selp.f32 $0, low, 0f00000000, finite;\n        }",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _tcgen05_fence_before_thread_sync(*, loc=None, ip=None) -> None:
    """Order this thread's prior tcgen05 operations before a following thread synchronization."""
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _tcgen05_fence_after_thread_sync(*, loc=None, ip=None) -> None:
    """Order this thread's following tcgen05 operations after a preceding thread synchronization."""
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


DQ_EPI_BATCH = 2


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
    SETUP_WARP = 19
    GATHER_THREADS = GATHER_WARPS * 32
    MATH_THREAD_BEGIN = MATH_WARP_BEGIN * 32
    MATH_THREADS = MATH_WARPS * 32
    REDUCE_THREAD_BEGIN = REDUCE_WARP_BEGIN * 32
    REDUCE_THREADS = REDUCE_WARPS * 32
    DKV_MMA_TILER = (256, 64, 64)
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
    # D-round-0 dO^T operand of the dV products (128 D lanes x 128 heads as BF16 pairs) when ``DV0_TMEM_A``.
    TMEM_A_OFFSET = 64
    DV0_TMEM_A = False
    # D-round-1 dO^T operand of the dV products, when ``DV1_TMEM_A`` (the dKV accumulator then holds one
    # D-round at a time in the round-0 columns, so these columns are free).
    TMEM_A1_OFFSET = 448
    DV1_TMEM_A = False
    SCORE_DONE_STAGES = 2
    ROUND_GENS_PER_TILE = 8
    ROUND_STAGES = 2
    MMA_DONE_STAGES = 2
    DQ_EPI_BATCH_CHUNKS = 4
    DQ_WIDE_STORE_VALUES = 8
    DQ_CONVERSION_PAIR_VALUES = 2
    GATHER_SETMAXREG = 88
    UTILITY_SETMAXREG = 96
    MATH_SETMAXREG = 120
    REDUCER_SETMAXREG = 88
    MATH_EPI_SETMAXREG = 200
    REDUCER_EPI_SETMAXREG = 24
    # Widest top-k whose per-wave dKV records stay in L2 with default eviction priority (see ``_deposit_rank_panel``).
    RANK_L2_RESIDENT_MAX_TOPK = 512
    # Widest top-k whose dK/dV ring sweeps walk the head chunks in a runtime loop (see ``_issue_dkv_sweep_rolled``);
    # two-tile (top-k 128) queries keep the unrolled sweeps.
    ROLLED_SWEEP_MAX_TOPK = 512
    KDQ_REFETCH_HALF = False
    # Narrowest top-k whose stationary Q/dO loads issue before the cluster rendezvous (see ``kernel``).
    EARLY_STATIONARY_MIN_TOPK = 512
    # Drains request the overflow-row offsets of all their indirect slots before resolving any (few-plane plans).
    RANK_DRAIN_OVERLAP_LOOKUPS = False
    # dV low-first plans (top-k >= ``DS2_MIN_TOPK``) keep the dS blocks of the dK products in their own buffer (``_use_ds2``).
    DS2_ENABLED = True
    # Widest top-k whose deterministic peer-half P_lo is staged in its own buffer (``p_lo_xchg``) instead of
    # ds_image: the dS image (dQ operand) then waits for neither the P_lo landing nor the sweeps.
    PLO_STAGING_MAX_TOPK = 0
    # The relay sends a tile's dS block to the peer as soon as the P_lo blocks are released.
    DS_RELAY_ON_RELEASE = False
    # Overflow records of wide waves keep default L2 priority for the following tail (few-plane plans).
    RANK_RECORDS_L2_DEFAULT = False
    # Dedicated-dS plans with both dO^T operands in TMEM sweep the D-rounds in alternating order per tile
    # (tile ``t`` takes D-round ``t % 2`` first): a tile's first Q^T group is the one its predecessor swept
    # last, still in the ring, so each tile after the first streams one Q^T group instead of two.
    ROUND_REUSE = False
    # Narrowest top-k that alternates its D-rounds: every dedicated-dS width.
    ROUND_REUSE_MIN_TOPK = 512
    # Narrowest top-k of the dedicated-dS plans (``_use_ds2``): four 8 KiB ring stages, both dO^T operands in TMEM.
    DS2_MIN_TOPK = 512
    deterministic = False
    rank_shard = False
    rank_fused = False

    def __init__(self, element_dtype: Type[cutlass.Numeric], head_dim: int, head_dim_v: int, block_tile: int, max_topk: int = 0):
        if element_dtype != BFloat16:
            raise ValueError(f"two-CTA DSA backward requires BF16, got {element_dtype}")
        if head_dim != 512 or head_dim_v != 512:
            raise ValueError("two-CTA DSA backward requires head_dim=head_dim_v=512")
        if block_tile != 64:
            raise ValueError(f"two-CTA DSA backward requires block_tile=64, got {block_tile}")
        if max_topk not in (128, 512, 1024, 1152, 2048):
            raise ValueError(f"two-CTA DSA backward requires max_topk in {{128, 512, 1024, 1152, 2048}}, got {max_topk}")
        self.element_dtype = element_dtype
        self.acc_dtype = Float32
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.head_dim_main = head_dim
        self.same_hdim_kv = True
        self.block_tile = block_tile
        self.max_topk = max_topk
        if self.deterministic:
            # Every top-k width refetches only the peer CTA's rows for the dQ operand (the own rows stay in place).
            self.KDQ_REFETCH_HALF = True
            # TMEM chunks the math warps load per wait in their dQ epilogue round: the 120-register math budget
            # of every top-k width holds a batch of DQ_EPI_BATCH chunks.
            self.DQ_EPI_MATH_BATCH = DQ_EPI_BATCH
            # dK/dV round stages: every accepted width streams four 8 KiB stages (32 heads each), which keeps
            # more refills in flight; the dedicated-dS plans (``_use_ds2``) rely on exactly four stages. The dV
            # passes interleave the high and residual-low P terms per stage, so dKV sums in a different (fixed)
            # order than with 64-head stages.
            self.ROUND_K_HEADS = 32
        else:
            self.KDQ_REFETCH_HALF = max_topk >= 512
            self.ROUND_K_HEADS = 64 if max_topk <= 512 else 32
        assert self.H_TILE_CLUSTER % self.ROUND_K_HEADS == 0
        self.ROUND_TILER = (self.D_TILE_CLUSTER, self.N_TILE, self.ROUND_K_HEADS)
        self.ROUND_STAGE_ELEMENTS = self.D_TILE_CTA * self.ROUND_K_HEADS
        self.ROUND_STAGE_BYTES = 2 * self.ROUND_STAGE_ELEMENTS
        self.ROUND_STAGES = self.ROUND_BUF_ELEMENTS // self.ROUND_STAGE_ELEMENTS
        self.ROUND_GENS_PER_TILE = 4 * (self.H_TILE_CLUSTER // self.ROUND_K_HEADS)
        assert self.ROUND_STAGES in (2, 4, 8)
        assert self.ROUND_GENS_PER_TILE % (2 * self.ROUND_STAGES) == 0
        if self.deterministic:
            # Deterministic plans set each role's budget inside its own branch (see ``kernel``).
            if self.max_topk >= 512:
                # The utility warpgroup (MMA, load, relay, setup warps) fits 80 registers; the math warps take
                # the released ones and stop spilling their softmax/dS state, the reducers keep 96.
                self.UTILITY_SETMAXREG = 80
                self.MATH_SETMAXREG = 120
                self.REDUCER_SETMAXREG = 96
        elif self.max_topk == 128:
            self.MATH_SETMAXREG = 104
            self.REDUCER_SETMAXREG = 96
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

    @cute.jit
    def rank_wave_span(self, total_q: Int32, num_waves: Int32, wave_size: Int32, wave: Int32, interleaved: Int32):
        """Return the first query and query count of wave ``wave`` (empty outside ``[0, num_waves)``).

        Consecutive waves hold queries ``[wave * wave_size, (wave + 1) * wave_size)``; interleaved
        waves (``interleaved != 0``) hold ``wave + j * num_waves``, so consecutive queries, whose
        top-k rows often overlap, fall into different waves.
        """
        begin = wave * wave_size
        size = max(min(wave_size, total_q - begin), Int32(0))
        if interleaved != Int32(0):
            begin = wave
            size = (total_q - wave + num_waves - Int32(1)) // num_waves
        if wave < Int32(0):
            begin = Int32(0)
            size = Int32(0)
        if wave >= num_waves:
            begin = Int32(0)
            size = Int32(0)
        return begin, size

    @cute.jit
    def _set_role_registers(self, count: cutlass.Constexpr[int]) -> None:
        """Move this warpgroup's register budget from the 96-register launch value to ``count``."""
        if cutlass.const_expr(count > 96):
            cute.arch.setmaxregister_increase(count)
        elif cutlass.const_expr(count < 96):
            cute.arch.setmaxregister_decrease(count)

    @cute.jit
    def _visit_tile(self, order: Int32, tile_count: Int32, ascending: Int32) -> Int32:
        """Return the tile visited ``order``-th: descending sweeps by default, ascending when ``ascending`` is 1.

        Non-deterministic plans always sweep descending (``ascending`` is unused).
        """
        if cutlass.const_expr(not self.deterministic):
            return tile_count - Int32(1) - order
        return order + (Int32(1) - ascending) * (tile_count - Int32(1) - Int32(2) * order)

    def _use_ds2(self) -> bool:
        """Whether the dK products read dS from their own blocks (``ds2_blocks``) instead of the P_lo blocks.

        Top-k >= ``DS2_MIN_TOPK`` plans (TMEM dO^T, four ring stages): the dS of a tile then reaches both CTAs
        without waiting for the dV products to release the P_lo blocks, and the peer half of P_hi is
        staged in the owner half of ``ds_image`` (its ``p_xchg`` buffer makes room for the dS blocks).
        """
        return bool(self.deterministic and self.DV0_TMEM_A and self.max_topk >= self.DS2_MIN_TOPK and self.DS2_ENABLED)

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
        slot and the k-block order is identical for every ROUND_K_HEADS. (The
        dV sweep interleaves the BF16 high and residual-low P passes per stage,
        so there the stage size does set the FP32 accumulation order.)
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
            b_k_half = head_base % self.H_TILE_CTA // self.ROUND_K_HEADS
            accumulate = True if chunk > 0 else first_accumulate
            round_pipeline.consumer_wait(round_consumer_state)
            self._issue_dkv_pass(dkv_tiled_mma, t_dkv, round_slot_fragments[chunk], b_fragment, b_k_half, accumulate)
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
        return round_consumer_state

    @cute.jit
    def _issue_dkv_sweep_rolled(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        round_staged: cute.Tensor,
        high_staged: cute.Tensor,
        low_staged: cute.Tensor,
        first_accumulate: cutlass.Constexpr[bool],
        use_low: cutlass.Constexpr[bool],
        wait_low_on_first_chunk: cutlass.Constexpr[bool],
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
    ):
        """``_issue_dkv_sweep`` / ``_issue_dkv_sweep_hilo`` as one loop trip per ring stage.

        The ring stage (A, its slot taken from the ring state) and the 64-head B block of every
        head chunk are indexed at run time, so the issue code holds one chunk's products; the
        products and their per-accumulator order are those of the unrolled sweeps.
        """
        k_blocks = self.ROUND_K_HEADS // 16
        gens = self.H_TILE_CTA // self.ROUND_K_HEADS
        for chunk in cutlass.range(0, self.ROUND_STAGES, unroll=1):
            b_stage = chunk // Int32(gens)
            b_k_base = chunk % Int32(gens) * Int32(k_blocks)
            slot = round_consumer_state.index
            round_pipeline.consumer_wait(round_consumer_state)
            mma = dkv_tiled_mma.with_()
            if cutlass.const_expr(first_accumulate):
                mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                mma.set(tcgen05.Field.ACCUMULATE, chunk > Int32(0))
            for k_block in cutlass.range_constexpr(k_blocks):
                cute.gemm(mma, t_dkv, round_staged[None, None, k_block, slot], high_staged[None, None, b_k_base + Int32(k_block), b_stage], t_dkv)
                mma.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(use_low):
                if cutlass.const_expr(wait_low_on_first_chunk):
                    if chunk == Int32(0):
                        pds_pipeline.consumer_wait(pds_consumer_state)
                mma = dkv_tiled_mma.with_()
                mma.set(tcgen05.Field.ACCUMULATE, True)
                for k_block in cutlass.range_constexpr(k_blocks):
                    cute.gemm(mma, t_dkv, round_staged[None, None, k_block, slot], low_staged[None, None, b_k_base + Int32(k_block), b_stage], t_dkv)
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
        return round_consumer_state

    @cute.jit
    def _issue_dkv_sweep_reuse(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        round_fragment_0: cute.Tensor,
        round_fragment_1: cute.Tensor,
        round_fragment_2: cute.Tensor,
        round_fragment_3: cute.Tensor,
        b_fragment_0: cute.Tensor,
        b_fragment_1: cute.Tensor,
        round_pipeline,
        generation_base: Int32,
        wait_full: cutlass.Boolean,
        release_empty: cutlass.Boolean,
    ):
        """``_issue_dkv_sweep`` (accumulating) over ring generations ``generation_base + chunk`` (``ROUND_REUSE``).

        ``wait_full`` waits for each stage to land (a group the ring already holds skips it) and
        ``release_empty`` frees each stage after its products (a group kept for the next tile skips it).
        """
        round_slot_fragments = (round_fragment_0, round_fragment_1, round_fragment_2, round_fragment_3)
        for chunk in cutlass.range_constexpr(self.ROUND_STAGES):
            head_base = chunk * self.ROUND_K_HEADS
            b_fragment = b_fragment_0 if head_base < self.H_TILE_CTA else b_fragment_1
            b_k_half = head_base % self.H_TILE_CTA // self.ROUND_K_HEADS
            generation = generation_base + Int32(chunk)
            stage_state = pipeline.PipelineState(self.ROUND_STAGES, generation, Int32(chunk), generation // Int32(self.ROUND_STAGES) & Int32(1))
            if wait_full:
                round_pipeline.consumer_wait(stage_state)
            self._issue_dkv_pass(dkv_tiled_mma, t_dkv, round_slot_fragments[chunk], b_fragment, b_k_half, True)
            if release_empty:
                round_pipeline.consumer_release(stage_state)

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
        if cutlass.const_expr(self.deterministic):
            # One loop trip per D128 chunk with its k-blocks unrolled: half the issue-loop code of the
            # unrolled-chunk form and one branch per eight products.
            for chunk in cutlass.range(0, self.K_CHUNKS, unroll=1):
                for k_block in cutlass.range_constexpr(k_blocks_per_chunk):
                    cute.gemm(mma, accumulator, a_fragment[None, None, k_block, chunk], b_fragment[None, None, k_block, chunk], accumulator)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
        else:
            for chunk in cutlass.range_constexpr(self.K_CHUNKS):
                for k_block in cutlass.range(0, k_blocks_per_chunk, unroll=4):
                    cute.gemm(mma, accumulator, a_fragment[None, None, k_block, chunk], b_fragment[None, None, k_block, chunk], accumulator)
                    mma.set(tcgen05.Field.ACCUMULATE, True)

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
        self, mLSE: cute.Tensor, mAttnSink: cute.Tensor, softmax_stats: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, reducer_tidx: Int32
    ) -> None:
        """Publish exact sink-folded negative LSE to shared memory."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        log2_e = Float32(math.log2(math.e))
        pos_inf = Float32(float("inf"))
        neg_inf = Float32(float("-inf"))
        if lane < Int32(8):
            row = row_base + lane
            head = rank * Int32(self.H_TILE_CTA) + row
            lse_value = Float32(mLSE[head, (token_idx, batch_idx)])
            sink_value = Float32(mAttnSink[head, (0, batch_idx)])
            lse_log2 = lse_value * log2_e
            sink_log2 = sink_value * log2_e
            neg_lse_log2 = neg_inf
            if lse_log2 != pos_inf and sink_log2 != pos_inf:
                if lse_log2 != neg_inf or sink_log2 != neg_inf:
                    maximum = cute.arch.fmax(lse_log2, sink_log2)
                    denominator = Float32(cute.math.exp2(lse_log2 - maximum) + cute.math.exp2(sink_log2 - maximum))
                    neg_lse_log2 = -(maximum + cute.math.log2(denominator))
            softmax_stats[row, 0] = neg_lse_log2

    @staticmethod
    def _get_workspace_size_LSE_OdO(q: int, d: int, h: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        q = (q + 7) // 8 * 8
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
        layout = cute.make_layout((num_heads, (total_q, 1)), stride=(1, (cute.assume(num_heads, divby=64), 0)))
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
        # dV round 0 with its dO^T operand resident in TMEM (A from TMEM is K-major: heads along columns).
        dkv_ts_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cg2,
            self.DKV_MMA_TILER[:2],
            tcgen05.OperandSource.TMEM,
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
        dq_epi_layout_staged = sm100_utils.make_smem_layout_epi(self.element_dtype, LayoutEnum.from_tensor(mdQ_epi), dq_epi_tile, 1)
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
        dq_epi_tile = sm100_utils.compute_epilogue_tile_shape(dq_cta_shape, True, LayoutEnum.ROW_MAJOR, self.acc_dtype)
        dq_tmem_load = sm100_utils.get_tmem_load_op(dq_cta_shape, LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, dq_epi_tile, True)
        sum_odo, scaled_lse = self._get_stats_workspace(workspace_LSE_OdO, mQ.shape[2][0], cute.size(problem_shape[3][0]))
        if cutlass.const_expr(self.deterministic):
            # Each query wave reuses the same bounded scratch shards. Launches on
            # this stream order successive waves before the final reduction.
            num_q_waves = cute.ceil_div(problem_shape[0], self.max_query_pairs_per_wave)
            q_wave_size = cute.ceil_div(problem_shape[0], num_q_waves)
            # Without per-query lengths every query spans the full top-k width, so conflict-rank plans
            # with enough dense planes interleave their waves (see ``rank_wave_span``). Plans with
            # lengths keep consecutive waves, whose similar lengths let short waves finish early,
            # unless fewer than half a wave of dense planes exist: then the rows that neighbouring
            # queries share would mostly overflow, and interleaving spreads those queries apart.
            interleaved = Int32(0)
            if cutlass.const_expr(self.rank_shard):
                _, interleave_planes = self.rank_geometry(mKV.shape[0])
                if interleave_planes >= Int32(RANK_INTERLEAVE_MIN_PLANES):
                    if cutlass.const_expr(mTopkLength is None):
                        interleaved = Int32(1)
                    elif Int32(2) * interleave_planes < Int32(self.max_query_pairs_per_wave):
                        interleaved = Int32(1)
                    elif Int32(16 * self.max_topk) <= problem_shape[0]:
                        # A top-k width small against the query count bounds any length ramp to a
                        # few waves, so neighbouring queries' shared rows are spread over waves too.
                        interleaved = Int32(1)
            query_stride = Int32(1)
            if interleaved != Int32(0):
                query_stride = num_q_waves
            if cutlass.const_expr(self.rank_shard):
                active_shards = Int32(1)
                mdKV_acc = self.make_rank_rows(workspace_dKV, mKV.shape[0])
                self.launch_rank_init(mdKV_acc, mKV.shape[0], stream)
                first_begin, first_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, Int32(0), interleaved)
                second_begin, second_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, Int32(1), interleaved)
                # Record wave 0's row ordinals. Fused plans also assign wave 0 and record wave 1 here,
                # after which each main kernel's setup warp prepares the following wave; per-wave-setup
                # plans run one setup launch before each main kernel (below).
                self.launch_rank_wave_setup(
                    mdKV_acc,
                    mTopkIdxs,
                    mTopkLength,
                    mKV.shape[0],
                    Int32(RANK_STEP_BIAS - 1),
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    first_begin,
                    first_size,
                    Int32(1),
                    query_stride,
                    stream,
                )
                if cutlass.const_expr(self.rank_fused):
                    self.launch_rank_wave_setup(
                        mdKV_acc,
                        mTopkIdxs,
                        mTopkLength,
                        mKV.shape[0],
                        Int32(RANK_STEP_BIAS),
                        Int32(0),
                        Int32(0),
                        first_begin,
                        first_size,
                        second_begin,
                        second_size,
                        Int32(1),
                        query_stride,
                        stream,
                    )
            else:
                active_shards = q_wave_size + min(q_wave_size, Int32(self.split_wg_queries_per_wave))
                mdKV_acc = self.make_dKV_accumulator(workspace_dKV, mKV.shape[0])
                self.zero_dKV_shards(mdKV_acc, mKV.shape[0]).launch(grid=[cute.ceil_div(mKV.shape[0], 4), active_shards, 1], block=[128, 1, 1], stream=stream)
        else:
            mdKV_acc = cute.make_tensor(cute.recast_ptr(workspace_dKV.iterator, dtype=self.acc_dtype), mdKV.layout)
            zero_grid_x = (mKV.shape[0] + self.ZERO_ROWS_PER_BLOCK - 1) // self.ZERO_ROWS_PER_BLOCK
            self.zero_init(mdKV_acc, mdSink, mKV.shape[0], cute.size(problem_shape[3][0])).launch(
                grid=[zero_grid_x, 1, problem_shape[3][1]], block=[self.ZERO_THREADS, 1, 1], stream=stream
            )
        if cutlass.const_expr(self.deterministic):
            for wave in cutlass.range(0, num_q_waves):
                token_begin, wave_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, wave, interleaved)
                previous_begin, previous_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, wave - Int32(1), interleaved)
                next_begin, next_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, wave + Int32(1), interleaved)
                after_begin, after_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, wave + Int32(2), interleaved)
                if cutlass.const_expr(self.rank_shard and not self.rank_fused):
                    self.launch_rank_wave_setup(
                        mdKV_acc,
                        mTopkIdxs,
                        mTopkLength,
                        mKV.shape[0],
                        wave + Int32(RANK_STEP_BIAS),
                        previous_begin,
                        previous_size,
                        token_begin,
                        wave_size,
                        next_begin,
                        next_size,
                        Int32(1) - min(wave, Int32(1)),
                        query_stride,
                        stream,
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
                    dkv_ts_tiled_mma,
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
                    token_begin,
                    wave + Int32(RANK_STEP_BIAS + 1),
                    token_begin,
                    wave_size,
                    next_begin,
                    next_size,
                    after_begin,
                    after_size,
                    wave + Int32(RANK_STEP_BIAS - 1),
                    previous_begin,
                    previous_size,
                    query_stride,
                ).launch(
                    grid=(2 * wave_size, 1, problem_shape[3][1]),
                    block=[self.THREADS_PER_CTA, 1, 1],
                    cluster=self.CLUSTER_SHAPE_MNK,
                    smem=self.shared_storage_bytes,
                    stream=stream,
                    min_blocks_per_mp=1,
                    use_pdl=self.rank_shard,
                )
        else:
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
                dkv_ts_tiled_mma,
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
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(1),
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
        if cutlass.const_expr(self.deterministic and self.rank_shard):
            # The last wave's tail runs with the dSink partials, then the used-plane fold.
            last_wave = num_q_waves - Int32(1)
            last_step = last_wave + Int32(RANK_STEP_BIAS)
            last_begin, last_size = self.rank_wave_span(problem_shape[0], num_q_waves, q_wave_size, last_wave, interleaved)
            if problem_shape[0] <= Int32(512):
                self.postprocess_rank(
                    mdKV_acc,
                    mdKV,
                    mKV.shape[0],
                    mTopkIdxs,
                    mTopkLength,
                    last_step,
                    last_begin,
                    last_size,
                    query_stride,
                    sum_odo,
                    scaled_lse,
                    mAttnSink,
                    mdSink,
                    problem_shape,
                    stream,
                    2,
                )
            elif problem_shape[0] <= Int32(2048):
                self.postprocess_rank(
                    mdKV_acc,
                    mdKV,
                    mKV.shape[0],
                    mTopkIdxs,
                    mTopkLength,
                    last_step,
                    last_begin,
                    last_size,
                    query_stride,
                    sum_odo,
                    scaled_lse,
                    mAttnSink,
                    mdSink,
                    problem_shape,
                    stream,
                    8,
                )
            else:
                self.postprocess_rank(
                    mdKV_acc,
                    mdKV,
                    mKV.shape[0],
                    mTopkIdxs,
                    mTopkLength,
                    last_step,
                    last_begin,
                    last_size,
                    query_stride,
                    sum_odo,
                    scaled_lse,
                    mAttnSink,
                    mdSink,
                    problem_shape,
                    stream,
                    32,
                )
        elif cutlass.const_expr(self.deterministic):
            if problem_shape[0] <= Int32(512):
                self.postprocess_deterministic(mdKV_acc, mdKV, mKV.shape[0], active_shards, sum_odo, scaled_lse, mAttnSink, mdSink, problem_shape, stream, 2)
            elif problem_shape[0] <= Int32(2048):
                self.postprocess_deterministic(mdKV_acc, mdKV, mKV.shape[0], active_shards, sum_odo, scaled_lse, mAttnSink, mdSink, problem_shape, stream, 8)
            else:
                self.postprocess_deterministic(mdKV_acc, mdKV, mKV.shape[0], active_shards, sum_odo, scaled_lse, mAttnSink, mdSink, problem_shape, stream, 32)
        else:
            self.convert_dkv(mdKV_acc, mdKV, mKV.shape[0]).launch(
                grid=[convert_grid_x, 1, 1], block=[self.num_threads_D_convert, self.num_threads_seq, 1], stream=stream
            )
        if cutlass.const_expr(self.deterministic):
            pass
        else:
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
        if cutlass.const_expr(self.deterministic):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                done_pipeline.consumer_release(release_state)
        else:
            done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
        if cutlass.const_expr(self.deterministic):
            kv_index = Int32(-1)
            owner_lane = Int32(0)
            for owner_wg in cutlass.range(2):
                if wg_idx == owner_wg:
                    for owner_lane in cutlass.range(4):
                        if dp_idx % Int32(4) == owner_lane:
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
                                    ordered_dkv_quad(target_frg_0, rdkv_frg_0)
                        cute.arch.sync_warp()
                # Owner 1 is ordered by the next disjoint-half/next-tile rendezvous.
                if owner_wg == Int32(0):
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
        if cutlass.const_expr(self.deterministic):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                done_pipeline.consumer_release(release_state)
        else:
            done_pipeline.consumer_release(release_state)
        release_state.advance()
        if cutlass.const_expr(self.deterministic):
            kv_index = Int32(-1)
            owner_lane = Int32(0)
            for owner_wg in cutlass.range(2):
                if wg_idx == owner_wg:
                    for owner_lane in cutlass.range(4):
                        if dp_idx % Int32(4) == owner_lane:
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
                                    ordered_dkv_quad(target_frg_1, rdkv_frg_1)
                        cute.arch.sync_warp()
                # The next tile rendezvous (or final CTA barrier) orders owner 1.
                if owner_wg == Int32(0):
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
    def _deposit_rank_panel(self, values: cute.Tensor, r_dest: cute.Tensor, r_dense: cute.Tensor, mRows: cute.Tensor, sub_tile_idx: Int32, dp_idx: Int32):
        """Add shared dense rows and store private overflow records for one D128 panel.

        Waves wider than ``RANK_L2_RESIDENT_MAX_TOPK`` mark their records L2 evict-first.
        """
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            if r_dest[i] >= Int32(0):
                dkv_row = mRows[None, r_dest[i], (0, 0)]
                tile_row = cute.flat_divide(dkv_row, (128,))
                tile_row = tile_row[None, sub_tile_idx]
                tile_row = cute.flat_divide(tile_row, (4,))
                target = tile_row[None, dp_idx // 4]
                if cutlass.const_expr(self.max_topk <= self.RANK_L2_RESIDENT_MAX_TOPK):
                    # A narrow wave's records fit in L2 next to its streamed operands, and the next wave
                    # touches the same plane rows: keep them resident (default eviction priority).
                    if r_dense[i] != Int32(0):
                        rank_reduce_f32x4(target.iterator, values[coord_base], values[coord_base + 2], values[coord_base + 16], values[coord_base + 18])
                    else:
                        rank_store_f32x4(target.iterator, values[coord_base], values[coord_base + 2], values[coord_base + 16], values[coord_base + 18])
                elif r_dense[i] != Int32(0):
                    rank_reduce_f32x4_stream(target.iterator, values[coord_base], values[coord_base + 2], values[coord_base + 16], values[coord_base + 18])
                elif cutlass.const_expr(self.RANK_RECORDS_L2_DEFAULT):
                    # Few-plane plans: the next setup launch's tail reads these records back right after the
                    # wave, so they keep default L2 priority (the dense plane-0 adds stay evict-first).
                    rank_store_f32x4(target.iterator, values[coord_base], values[coord_base + 2], values[coord_base + 16], values[coord_base + 18])
                else:
                    rank_store_f32x4_stream(target.iterator, values[coord_base], values[coord_base + 2], values[coord_base + 16], values[coord_base + 18])

    @cute.jit
    def _drain_dkv_rank(
        self,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        mRows: cute.Tensor,
        index_row: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        seqlen_kv: Int32,
        rank_pitch: Int64,
        rank_planes: Int32,
        parity: Int32,
        ordinal: Int32,
        rtx: Int32,
        rank: Int32,
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
        offsets_base: Optional[cute.Pointer] = None,
        record_base: Optional[Int32] = None,
        round_swap: Int32 = 0,
    ):
        """Drain FP32 TMEM accumulators to conflict-rank destination records.

        ``round_swap`` 1 marks a tile whose first accumulator round holds D-round 1 (``ROUND_REUSE``).

        Each destination record has one writer in the wave, so warp groups and
        lane groups write concurrently.
        """
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
        r_dest = cute.make_rmem_tensor((8,), cutlass.Int32)
        r_dense = cute.make_rmem_tensor((8,), cutlass.Int32)
        r_row = cute.make_rmem_tensor((8,), cutlass.Int32)
        r_code = cute.make_rmem_tensor((8,), cutlass.Int32)
        # Every tile slot is below the top-k width, so the index and code loads of all eight slots
        # issue together; only the overflow offsets depend on them.
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            slot = tile_base + Int32(cute.get(thread_coordinates[coord_base], mode=[1]))
            r_row[i] = index_row[slot]
            r_code[i] = self.rank_code(mRows, rank_pitch, rank_planes, parity, ordinal, slot)
        if cutlass.const_expr(self.RANK_DRAIN_OVERLAP_LOOKUPS):
            # Overflow codes name the j-th record of their row. Every indirect slot requests its row's offset
            # (into its no longer needed row register) before any offset is used, so the dependent lookups
            # of a tile overlap instead of following each other; destinations match ``rank_code_destination``.
            # The per-kernel offsets and record bases come from the caller (computed once per kernel).
            any_indirect = Int32(0)
            for i in cutlass.range_constexpr(8):
                coord_base = i * 2 - i % 2
                slot = tile_base + Int32(cute.get(thread_coordinates[coord_base], mode=[1]))
                row = r_row[i]
                code = r_code[i]
                r_dest[i] = Int32(-1)
                r_dense[i] = Int32(0)
                if slot < topk and row >= Int32(0) and row < seqlen_kv:
                    r_dest[i] = code & Int32(RANK_CODE_VALUE)
                    if (code & Int32(RANK_CODE_STORE)) == Int32(0):
                        r_dense[i] = Int32(1)
                    if (code & Int32(RANK_CODE_INDIRECT)) != Int32(0):
                        r_row[i] = cute.arch.load(offsets_base + Int64(row), Int32)
                        any_indirect = Int32(1)
            if any_indirect != Int32(0):
                for i in cutlass.range_constexpr(8):
                    if r_dest[i] >= Int32(0) and (r_code[i] & Int32(RANK_CODE_INDIRECT)) != Int32(0):
                        r_dest[i] = record_base + r_row[i] + r_dest[i]
        else:
            for i in cutlass.range_constexpr(8):
                coord_base = i * 2 - i % 2
                slot = tile_base + Int32(cute.get(thread_coordinates[coord_base], mode=[1]))
                row = r_row[i]
                r_dest[i] = Int32(-1)
                r_dense[i] = Int32(0)
                if slot < topk and row >= Int32(0) and row < seqlen_kv:
                    destination, dense = self.rank_code_destination(mRows, rank_pitch, rank_planes, parity, r_code[i], row)
                    r_dest[i] = destination
                    if dense:
                        r_dense[i] = Int32(1)
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        self._deposit_rank_panel(thread_values_0, r_dest, r_dense, mRows, Int32(2) * round_swap + rank, dp_idx)
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            done_pipeline.consumer_release(release_state)
        release_state.advance()
        self._deposit_rank_panel(thread_values_1, r_dest, r_dense, mRows, Int32(2) - Int32(2) * round_swap + rank, dp_idx)
        return (wait_state, release_state)

    @cute.jit
    def _load_o_bf16x16(self, source: cute.Pointer):
        """Load immutable O: read-once (L1 bypass, L2 evict-first) when deterministic, else policy-neutral."""
        if cutlass.const_expr(self.deterministic):
            return _load_o_bf16x16_read_once(source)
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
        values = (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)
        return _dot_f32x16(*values)

    @cute.jit
    def _dot_odo_bf16x16_bits(self, out_bits: U64x4, dout_bits: U64x4) -> Float32:
        """Decode one packed segment pair and retain the fixed FMA order."""
        out_values = _decode_bf16x16_to_f32(out_bits[0], out_bits[1], out_bits[2], out_bits[3])
        dout_values = _decode_bf16x16_to_f32(dout_bits[0], dout_bits[1], dout_bits[2], dout_bits[3])
        return self._dot_odo_f32x16(*out_values, *dout_values)

    @cute.kernel
    def convert_dkv(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen: Int32):
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
            values = (*_load_f32x4(source), *_load_f32x4(source + Int32(4)), *_load_f32x4(source + Int32(8)), *_load_f32x4(source + Int32(12)))
            chunk = source_base // Int32(32) * Int32(32)
            a0 = source_base % Int32(32) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = _dq_pack_bf16x2(values[b], values[b + 4])
                packed23 = _dq_pack_bf16x2(values[b + 8], values[b + 12])
                _store_bf16x4_ordinary(out_row.iterator + output_base, packed01, packed23)

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
                destination_rows = cute.composition(destination[None, None, None, chunk], cute.make_layout((self.N_TILE_CTA, self.K_CHUNK)))
                if kv_index >= 0 and kv_index < seqlen_kv:
                    self._copy_sparse_k_d128_row(
                        mKV, destination_rows, local_n, kv_index, batch_idx, Int32(chunk * self.K_CHUNK), index_in_group, copy_atom, thread_copy
                    )
                else:
                    self._zero_sparse_k_d128_row(destination_rows, local_n, index_in_group)

    @cute.jit
    def _load_prefetch_row(
        self,
        mTopkIdxs: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        order: Int32,
        tile_count: Int32,
        topk: Int32,
        rank: Int32,
        lane: Int32,
        ascending: Int32 = 0,
    ) -> Int32:
        """Return this lane's rank-owned KV row of the tile visited ``order``-th (see ``_visit_tile``), or -1."""
        row = Int32(-1)
        if order < tile_count:
            slot = self._visit_tile(order, tile_count, ascending) * Int32(self.N_TILE) + rank * Int32(self.N_TILE_CTA) + lane
            if slot < topk:
                row = mTopkIdxs[slot, (token_idx, batch_idx)]
        return row

    @cute.jit
    def _prefetch_kv_row_l2(self, mKV: cute.Tensor, row: Int32, batch_idx: Int32) -> None:
        """Hint one full D512 KV row into L2 ahead of its score gather (both K-dQ halves read it too)."""
        if row >= Int32(0) and row < cute.size(mKV, mode=[0]):
            _prefetch_bulk_l2(mKV.iterator + mKV.layout((row, Int32(0), (Int32(0), batch_idx))), self.D_HEAD * 2)

    @cute.jit
    def _prefetch_odo_rows(self, mOut: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, reducer_tidx: Int32) -> None:
        """Warm every O chunk this reducer thread will read before the LSE fold."""
        stats_warp = reducer_tidx // Int32(32)
        lane = reducer_tidx % Int32(32)
        row_base = stats_warp * Int32(8)
        row = row_base + lane // Int32(4)
        group_lane = lane % Int32(4)
        head = rank * Int32(self.H_TILE_CTA) + row
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
    def _load_stationary_do_bf16x16(self, stationary_do_physical: cute.Tensor, row: Int32, dim: Int32) -> U64x4:
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
        for chunk in cutlass.range_constexpr(4, 8):
            logical_old_lane = group_lane + Int32(4 * chunk)
            dim = logical_old_lane * Int32(16)
            out_pointer = mOut.iterator + mOut.layout((head, dim, (token_idx, batch_idx)))
            out_bits = self._load_o_bf16x16(out_pointer)
            dout_bits = self._load_stationary_do_bf16x16(stationary_do_physical, row, dim)
            partials[chunk] = self._dot_odo_bf16x16_bits(out_bits, dout_bits)
        owned_row_sum = _reduce_s8_dim_chunks_owned(partials[0], partials[1], partials[2], partials[3], partials[4], partials[5], partials[6], partials[7])
        if group_lane == Int32(0):
            raw_neg_sum_odo = -owned_row_sum
            softmax_stats[row, 1] = raw_neg_sum_odo

    @cute.jit
    def _publish_dsink_stats(
        self, sum_odo: cute.Tensor, scaled_lse: cute.Tensor, softmax_stats: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, reducer_tidx: Int32
    ) -> None:
        if reducer_tidx < Int32(self.H_TILE_CTA):
            row = reducer_tidx
            head = rank * Int32(self.H_TILE_CTA) + row
            sum_odo[head, (token_idx, batch_idx)] = softmax_stats[row, 1]
            scaled_lse[head, (token_idx, batch_idx)] = softmax_stats[row, 0]

    @cute.kernel
    def zero_init(self, mdKV_acc: cute.Tensor, mdSink: cute.Tensor, seqlen_kv: Int32, num_heads: Int32):
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
    def sum_dSink(self, sum_odo: cute.Tensor, scaled_lse: cute.Tensor, attn_sink: cute.Tensor, d_sink: cute.Tensor, problem_shape):
        q_block_idx, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        head_idx = tidx
        q_end = min(problem_shape[0], (q_block_idx + 1) * self.DSINK_BLOCK_Q)
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
        dkv_ts_tiled_mma: cute.TiledMma,
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
        token_begin: Int32,
        setup_step: Int32,
        setup_prev_begin: Int32,
        setup_prev_size: Int32,
        setup_cur_begin: Int32,
        setup_cur_size: Int32,
        setup_next_begin: Int32,
        setup_next_size: Int32,
        tail_step: Int32,
        tail_begin: Int32,
        tail_size: Int32,
        query_stride: Int32,
    ):
        """Execute the FP32 five-GEMM two-CTA schedule.

        Fused conflict-rank plans also run the next wave's setup phases on the
        otherwise idle warp 19 (``setup_*`` describe that pass), and the reducer
        warps add the previous wave's overflow records (``tail_*``) to plane 0
        before draining; that wave's setup moved this wave's ranks of those rows
        off plane 0.
        """
        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        use_ds2 = self._use_ds2()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)
        peer_rank = Int32(1) - rank
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        if cutlass.const_expr(self.deterministic):
            # Wave ordinal ``query_shard`` is query ``token_begin + query_shard * query_stride``.
            query_shard = token_idx
            token_idx = token_begin + query_shard * query_stride
            if cutlass.const_expr(self.rank_shard):
                rank_parity = (tail_step + Int32(1)) % Int32(2)
                split_wg_shard = cutlass.Boolean(False)
            else:
                mdKV_acc, split_wg_shard = self.select_dKV_accumulator(mdKV_acc, mKV.shape[0], query_shard, rank)
        is_leader_cta = rank == Int32(0)
        # Conflict-rank waves alternate the tile sweep direction: the plane rows a wave deposits last are
        # the rows the next wave deposits first, while they are still in L2.
        # A K <= 128 wave's records fit in L2 whatever the order, so those waves keep descending sweeps; so do
        # waves wider than RANK_L2_RESIDENT_MAX_TOPK, whose per-wave records far exceed L2 (evict-first deposits).
        # Non-deterministic plans always sweep descending (``_visit_tile``).
        sweep_ascending = Int32(0) if cutlass.const_expr(self.deterministic) else None
        if cutlass.const_expr(self.deterministic and self.rank_shard and self.max_topk > 128 and self.max_topk <= self.RANK_L2_RESIDENT_MAX_TOPK):
            sweep_ascending = (tail_step + Int32(1)) % Int32(2)
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
        if cutlass.const_expr(self.deterministic):
            ds_stage_free_mbar = storage.ds_stage_free_mbar.data_ptr()
            dsq_ready_mbar = storage.dsq_ready_mbar.data_ptr()
        plo_staging = self.deterministic and self.max_topk <= self.PLO_STAGING_MAX_TOPK
        p_lo_xchg_ptr = storage.p_lo_xchg.data_ptr() if plo_staging else None
        stationary_q_raw = storage.stationary_q.data_ptr()
        stationary_do_raw = storage.stationary_do.data_ptr()
        round_buf_raw = storage.round_buf.data_ptr()
        round_slot_raw = tuple((round_buf_raw + slot * self.ROUND_STAGE_ELEMENTS for slot in range(self.ROUND_STAGES)))
        score_kv_raw = storage.score_kv.data_ptr()
        stationary_q = storage.stationary_q.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_do = storage.stationary_do.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_q_tma = storage.stationary_q.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        stationary_do_tma = storage.stationary_do.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        stationary_do_physical = cute.make_tensor(stationary_do_raw, cute.make_layout((self.H_TILE_CTA * self.D_HEAD,), stride=(1,)))
        stationary_do_rows = cute.composition(stationary_do_tma[None, None, None, 0], cute.make_layout((self.H_TILE_CTA, self.D_HEAD), stride=(self.D_HEAD, 1)))
        k_n = storage.score_kv.get_tensor(score_b_layout_staged.outer, swizzle=score_b_layout_staged.inner)
        s_dq_epi = cute.make_tensor(cute.recast_ptr(storage.score_kv.data_ptr(), dq_epi_layout_staged.inner, self.element_dtype), dq_epi_layout_staged.outer)[
            None, None, 0
        ]
        kdq_loan_ptr_0 = cute.make_ptr(self.element_dtype, score_kv_raw.toint(), score_kv_raw.memspace, assumed_align=1024)
        kdq_loan_ptr_1 = cute.make_ptr(self.element_dtype, score_kv_raw.toint() + Int32(16384), score_kv_raw.memspace, assumed_align=1024)
        if cutlass.const_expr(self.KDQ_REFETCH_HALF):
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
        else:
            kdq_outer = dq_a_layout_staged.outer
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
        ds2_blocks = None
        ds2_block_raw_ptrs = None
        if cutlass.const_expr(use_ds2):
            # The P_hi half sent to the peer is staged in the owner half of ds_image (P_lo's staging
            # takes the peer half): both are rewritten only after the peer's P_lo landing (ds_stage_free).
            p_xchg_raw = cute.make_tensor(ds_image_raw + rank * Int32(self.PDS_BLOCK_ELEMENTS), flat_pds_block_layout)
            ds2_blocks_raw = storage.ds2_blocks.data_ptr()
            ds2_blocks = (
                cute.make_tensor(cute.recast_ptr(ds2_blocks_raw, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer),
                cute.make_tensor(
                    cute.recast_ptr(ds2_blocks_raw + self.PDS_BLOCK_ELEMENTS, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer
                ),
            )
            ds2_block_raw_ptrs = (ds2_blocks_raw, ds2_blocks_raw + self.PDS_BLOCK_ELEMENTS)
        else:
            p_xchg_raw = storage.p_xchg.get_tensor(flat_pds_block_layout)
        softmax_stats = storage.stats.get_tensor(cute.make_layout((self.H_TILE_CTA, 2), stride=(1, self.H_TILE_CTA)))
        staged_indices = storage.tile_indices.get_tensor(cute.make_layout((self.N_TILE, 2), stride=(1, self.N_TILE)))
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
        ds2_fragments = (None, None)
        if cutlass.const_expr(use_ds2):
            ds2_fragments = (dkv_tiled_mma.make_fragment_B(ds2_blocks[0]), dkv_tiled_mma.make_fragment_B(ds2_blocks[1]))
        rolled_fragments = None
        if cutlass.const_expr(self.deterministic and self.max_topk > 128 and self.max_topk <= self.ROLLED_SWEEP_MAX_TOPK):
            # Stage-indexed views of the contiguous ring stages and of the two P / dS blocks (same swizzle
            # phase: every stage starts on a 1 KiB boundary) for the rolled dK/dV sweeps.
            round_ring_layout = sm100_utils.make_smem_layout_a(dkv_tiled_mma, self.ROUND_TILER, self.element_dtype, self.ROUND_STAGES)
            assert cute.cosize(round_ring_layout) == self.ROUND_STAGES * self.ROUND_STAGE_ELEMENTS
            pds_pair_layout = sm100_utils.make_smem_layout_b(dkv_tiled_mma, self.DKV_MMA_TILER, self.element_dtype, 2)
            assert cute.cosize(pds_pair_layout) == 2 * self.PDS_BLOCK_ELEMENTS
            round_ring = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.element_dtype, round_buf_raw.toint(), round_buf_raw.memspace, assumed_align=1024),
                    round_ring_layout.inner,
                    dtype=self.element_dtype,
                ),
                round_ring_layout.outer,
            )
            p_pair = cute.make_tensor(cute.recast_ptr(p_blocks_raw, pds_pair_layout.inner, dtype=self.element_dtype), pds_pair_layout.outer)
            ds_pair = cute.make_tensor(cute.recast_ptr(ds_blocks_raw, pds_pair_layout.inner, dtype=self.element_dtype), pds_pair_layout.outer)
            rolled_fragments = (
                dkv_tiled_mma.make_fragment_A(round_ring),
                dkv_tiled_mma.make_fragment_B(p_pair),
                dkv_tiled_mma.make_fragment_B(ds_pair),
            )
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
        if cutlass.const_expr(self.deterministic):
            # The score/dP and dKV accumulators are released by one elected lane per consumer warp (after the warp's
            # TMEM loads completed), so each release costs a few mbarrier arrivals instead of one per thread.
            math_warp_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.MATH_WARPS)
            reduce_warp_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.REDUCE_WARPS)
        else:
            reduce_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.REDUCE_THREADS)
        pipe_s_done = self._make_umma_async_pipeline(
            self.SCORE_DONE_STAGES,
            leader_group,
            math_warp_group if cutlass.const_expr(self.deterministic) else math_group,
            storage.s_done_mbars.data_ptr(),
            cluster_layout_vmnk,
        )
        pipe_dp_done = self._make_umma_async_pipeline(
            self.SCORE_DONE_STAGES,
            leader_group,
            math_warp_group if cutlass.const_expr(self.deterministic) else math_group,
            storage.dp_done_mbars.data_ptr(),
            cluster_layout_vmnk,
        )
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
        pds_commit_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size)
        pipe_pds = self._make_async_umma_pipeline(1, pds_commit_group, leader_group, storage.pds_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_ds2 = None
        if cutlass.const_expr(use_ds2):
            pipe_ds2 = self._make_async_umma_pipeline(1, pds_commit_group, leader_group, storage.ds2_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_dkv_done = self._make_umma_async_pipeline(
            self.MMA_DONE_STAGES,
            leader_group,
            reduce_warp_group if cutlass.const_expr(self.deterministic) else reduce_group,
            storage.dkv_done_mbars.data_ptr(),
            cluster_layout_vmnk,
        )
        pipe_dq_done = self._make_umma_async_pipeline(1, leader_group, math_group, storage.dq_done_mbars.data_ptr(), cluster_layout_vmnk)
        pipe_dq0_done = self._make_umma_async_pipeline(1, leader_group, gather_group, storage.dq0_done_mbars.data_ptr(), cluster_layout_vmnk)
        early_stationary = self.deterministic and self.max_topk >= self.EARLY_STATIONARY_MIN_TOPK
        if cutlass.const_expr(early_stationary):
            if warp_idx == Int32(self.LOAD_WARP):
                # The stationary Q/dO loads depend on nothing but the token: the load warp initializes their
                # CTA-local barriers itself and issues both before the cluster rendezvous, so they land while
                # thread 0 initializes the remaining barriers.
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(stationary_tma_mbars, 1)
                    cute.arch.mbarrier_init(stationary_tma_mbars + 1, 1)
                    cute.arch.mbarrier_init_fence()
                cute.arch.fence_view_async_shared()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                    cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                cute.copy(tma_atom_q, t_q_gmem[None, rank, 0], t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                cute.copy(tma_atom_do, t_do_gmem[None, rank, 0], t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
        if tidx == Int32(0):
            if cutlass.const_expr(not early_stationary):
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
            if cutlass.const_expr(use_ds2):
                # Dedicated-dS plans: every math warp of both CTAs reports its dS image directly.
                cute.arch.mbarrier_init(ds_local_ready_mbar, 2 * self.MATH_WARPS)
            else:
                cute.arch.mbarrier_init(ds_local_ready_mbar, 2)
            if cutlass.const_expr(self.deterministic):
                cute.arch.mbarrier_init(ds_stage_free_mbar, 1)
                cute.arch.mbarrier_init(dsq_ready_mbar, self.MATH_WARPS)
            if cutlass.const_expr(self.DV0_TMEM_A):
                # One arrival per CTA once its math warps stored the D-round-0 dO^T lanes.
                cute.arch.mbarrier_init(storage.dot_tmem_ready_mbar.data_ptr(), 2)
        cute.arch.fence_view_async_shared()
        # The cluster rendezvous only publishes the barrier initializations (fence.mbarrier_init inside the
        # arrive helper): a relaxed arrive does not wait for this thread's in-flight global loads (the
        # top-k length above), which a release arrive would drain first in every warp of both CTAs.
        # Non-deterministic plans keep the release arrive.
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=self.deterministic)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = TmemAllocator(
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
        dot_tmem_fragment = None
        t_dot_fill = None
        if cutlass.const_expr(self.DV0_TMEM_A):
            dot_tmem_layout = cute.slice_(
                sm100_utils.make_smem_layout_a(dkv_ts_tiled_mma, (self.D_TILE_CLUSTER, self.N_TILE, self.H_TILE_CLUSTER), self.element_dtype, 1),
                (None, None, None, 0),
            )
            t_dot_tmem = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.TMEM_A_OFFSET, dtype=self.element_dtype), dot_tmem_layout.outer)
            # make_fragment_A addresses a TMEM operand from column 0: keep its layout on this operand's own columns.
            dot_tmem_fragment = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.TMEM_A_OFFSET, dtype=self.element_dtype), dkv_ts_tiled_mma.make_fragment_A(t_dot_tmem).layout
            )
            # FP32 view of the same columns (lane = D row of this CTA, column = head pair) for the fill.
            t_dot_fill = cute.make_tensor(tmem_ptr + self.TMEM_A_OFFSET, dkv_c_layout)
        dot_tmem_fragment_1 = None
        t_dot_fill_1 = None
        if cutlass.const_expr(self.DV1_TMEM_A):
            t_dot_tmem_1 = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.TMEM_A1_OFFSET, dtype=self.element_dtype), dot_tmem_layout.outer)
            dot_tmem_fragment_1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.TMEM_A1_OFFSET, dtype=self.element_dtype), dkv_ts_tiled_mma.make_fragment_A(t_dot_tmem_1).layout
            )
            t_dot_fill_1 = cute.make_tensor(tmem_ptr + self.TMEM_A1_OFFSET, dkv_c_layout)
        topk = raw_topk
        if topk > Int32(mTopkIdxs.shape[0]):
            topk = Int32(mTopkIdxs.shape[0])
        if topk < Int32(0):
            topk = Int32(0)
        tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(self.N_TILE)
        tile_count = cute.arch.make_warp_uniform(tile_count)
        # Register budgets: non-deterministic plans set every role's budget in this leading branch chain.
        # Deterministic plans set each role's budget at the top of its own branch below (the utility
        # warpgroup before its role split): ptxas sizes each code region for the budget that dominates
        # it, so a budget set in the leading chain caps every role region at the smallest one (88).
        if cutlass.const_expr(not self.deterministic):
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
            elif cutlass.const_expr(self.REDUCER_SETMAXREG > 96):
                cute.arch.setmaxregister_increase(self.REDUCER_SETMAXREG)
            elif cutlass.const_expr(self.REDUCER_SETMAXREG < 96):
                cute.arch.setmaxregister_decrease(self.REDUCER_SETMAXREG)
        if warp_idx < Int32(self.GATHER_WARPS):
            if cutlass.const_expr(self.deterministic):
                self._set_role_registers(self.GATHER_SETMAXREG)
            gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dq0_done_gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            gather_kd_rows_0 = self._kd_round_rows(kdq_loan[0])
            gather_kd_rows_1 = self._kd_round_rows(kdq_loan[1])
            if tile_count > Int32(0):
                if cutlass.const_expr(self.deterministic):
                    # Gather warp 0 keeps the rank-owned rows of a later tile warming in L2 (row ids load early, hints issue later).
                    pf_row = Int32(-1)
                    if tidx < Int32(32):
                        pf_row = self._load_prefetch_row(mTopkIdxs, token_idx, batch_idx, Int32(1), tile_count, topk, rank, tidx, sweep_ascending)
                self._prefetch_tile_indices(
                    mTopkIdxs, staged_indices, token_idx, batch_idx, self._visit_tile(Int32(0), tile_count, sweep_ascending), Int32(0), tidx
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
                    self._visit_tile(Int32(0), tile_count, sweep_ascending),
                    topk,
                    rank,
                    tidx,
                    kv_copy_atom,
                    kv_thread_copy,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                if tidx & Int32(31) == Int32(0):
                    pipe_kscore.producer_commit(gather_state)
                gather_state.advance()
                if cutlass.const_expr(self.deterministic):
                    if tidx < Int32(32):
                        self._prefetch_kv_row_l2(mKV, pf_row, batch_idx)
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
                            self._visit_tile(next_score_iter, tile_count, sweep_ascending),
                            next_index_slot,
                            tidx,
                        )
                    if cutlass.const_expr(self.deterministic):
                        pf_row = Int32(-1)
                        if tidx < Int32(32):
                            pf_row = self._load_prefetch_row(
                                mTopkIdxs, token_idx, batch_idx, score_iter + Int32(2), tile_count, topk, rank, tidx, sweep_ascending
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
                        self._visit_tile(score_iter, tile_count, sweep_ascending),
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    if tidx & Int32(31) == Int32(0):
                        pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                    if score_iter != tile_count - Int32(1):
                        if cutlass.const_expr(self.deterministic):
                            if tidx < Int32(32):
                                self._prefetch_kv_row_l2(mKV, pf_row, batch_idx)
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
                            self._visit_tile(next_iter, tile_count, sweep_ascending),
                            topk,
                            rank,
                            tidx,
                            kv_copy_atom,
                            kv_thread_copy,
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        if tidx & Int32(31) == Int32(0):
                            pipe_kscore.producer_commit(gather_state)
                        gather_state.advance()
                pipe_dq0_done.consumer_wait(dq0_done_gather_state)
                self._store_dq_epi_scalar_direct(
                    t_dq[0], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, mdQ, 0, token_idx, batch_idx, rank, tidx
                )
                pipe_dq0_done.consumer_release(dq0_done_gather_state)
                dq0_done_gather_state.advance()
                pipe_kscore.producer_tail(gather_state)
        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            if cutlass.const_expr(self.deterministic):
                self._set_role_registers(self.MATH_SETMAXREG)
            mtx = tidx - Int32(self.MATH_THREAD_BEGIN)
            if cutlass.const_expr(self.DV0_TMEM_A):
                if tile_count > Int32(0):
                    self._fill_dot_tmem(t_dot_fill, mdO, token_idx, batch_idx, rank, mtx)
                    if cutlass.const_expr(self.DV1_TMEM_A):
                        self._fill_dot_tmem(t_dot_fill_1, mdO, token_idx, batch_idx, rank, mtx, self.D_TILE_CLUSTER)
                    _partial_cta_sync(self.math_barrier.barrier_id, self.math_barrier.num_threads)
                    if mtx == Int32(0):
                        cute.arch.mbarrier_arrive(storage.dot_tmem_ready_mbar.data_ptr(), Int32(0))
            if tile_count > Int32(0):
                self.stats_lse_barrier.arrive_and_wait()
            else:
                self.math_barrier.arrive_and_wait()
            s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            dp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            pds_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            ds2_state = None
            if cutlass.const_expr(use_ds2):
                ds2_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
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
            # The non-owner warps' P_lo staging: ds_image, or with ``plo_staging`` its own buffer.
            t_rs_p_lo_xchg_tile = t_rs_ds_tile
            if cutlass.const_expr(plo_staging):
                aligned_p_lo_xchg_ptr = cute.make_ptr(
                    self.element_dtype,
                    p_lo_xchg_ptr.toint() - mtx // Int32(self.H_TILE_CTA) * Int32(self.PDS_BLOCK_BYTES),
                    p_lo_xchg_ptr.memspace,
                    assumed_align=16,
                )
                p_lo_xchg_store = cute.make_tensor(
                    cute.recast_ptr(aligned_p_lo_xchg_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain
                )
                t_rs_p_lo_xchg = thread_copy_r2s.partition_D(p_lo_xchg_store)
                assert cute.size(t_rs_p_lo_xchg, mode=[4]) == 1
                t_rs_p_lo_xchg_tile = t_rs_p_lo_xchg[None, None, None, None, 0]
            t_rs_ds2_local_tile = None
            if cutlass.const_expr(use_ds2):
                aligned_ds2_blocks_ptr = cute.make_ptr(self.element_dtype, ds2_blocks[0].iterator.toint(), ds2_blocks[0].memspace, assumed_align=16)
                ds2_local_store = cute.make_tensor(
                    cute.recast_ptr(aligned_ds2_blocks_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain
                )
                t_rs_ds2_local = thread_copy_r2s.partition_D(ds2_local_store)
                assert cute.size(t_rs_ds2_local, mode=[4]) == 1
                t_rs_ds2_local_tile = t_rs_ds2_local[None, None, None, None, 0]
            r_score = cute.make_rmem_tensor(score_coordinates.shape, self.acc_dtype)
            r_dp = cute.make_rmem_tensor(score_coordinates.shape, self.acc_dtype)
            r_p = cute.make_rmem_tensor(score_coordinates.shape, self.element_dtype)
            if cutlass.const_expr(self.deterministic):
                # Adjacent band pairs (i0 even, i1 = i0 + 1) share one BF16x2 word: one packed RN conversion each.
                r_p_words = cute.recast_tensor(r_p, cutlass.Uint32)
            r_valid_band = cute.make_rmem_tensor((8,), cutlass.Boolean)
            r_ds = r_p
            r_delta = cute.make_rmem_tensor((4,), self.acc_dtype)
            softmax_scale_log2_e = scale_softmax * Float32(math.log2(math.e))
            hoist_group_bases = [2 * (h_group % 2) + 16 * (h_group // 2) for h_group in range(4)]
            hoist_group_local_h = [Int32(cute.get(score_coordinates[group_base], mode=[0])) % Int32(self.H_TILE_CTA) for group_base in hoist_group_bases]
            hoist_band_indices = [[group_base + j % 2 + 4 * (j // 2) for j in range(8)] for group_base in hoist_group_bases]
            hoist_band_n = [Int32(cute.get(score_coordinates[hoist_band_indices[0][band]], mode=[1])) for band in range(8)]
            hoist_lse = [softmax_stats[hoist_group_local_h[h_group], 0] for h_group in range(4)]
            math_owns_local_half = cute.arch.make_warp_uniform(mtx // Int32(self.H_TILE_CTA)) == rank
            if cutlass.const_expr(self.deterministic):
                math_index_row = mTopkIdxs[None, (token_idx, batch_idx)]
            for loop_iter in cutlass.range(tile_count):
                if cutlass.const_expr(self.deterministic):
                    # Score tiles visit the sparse row in the wave's sweep order. The band row ids come from the
                    # immutable index row in global memory (issued before the score wait), so the math
                    # warps never read the gather warps' cp.async index staging.
                    tile_index = self._visit_tile(loop_iter, tile_count, sweep_ascending)
                    for band in cutlass.range_constexpr(8):
                        slot = tile_index * Int32(self.N_TILE) + hoist_band_n[band]
                        r_valid_band[band] = False
                        if slot < topk:
                            row = math_index_row[slot]
                            r_valid_band[band] = row >= Int32(0) and row < mKV.shape[0]
                pipe_s_done.consumer_wait(s_state)
                if s_state.index == Int32(0):
                    cute.copy(score_copy, score_source, r_score)
                else:
                    cute.copy(score_copy_pp, score_source_pp, r_score)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.deterministic):
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipe_s_done.consumer_release(s_state)
                else:
                    pipe_s_done.consumer_release(s_state)
                s_state.advance()
                if cutlass.const_expr(not self.deterministic):
                    # Score tiles visit the sparse row in reverse order.
                    tile_index = tile_count - Int32(1) - loop_iter
                    for band in cutlass.range_constexpr(8):
                        slot = tile_index * Int32(self.N_TILE) + hoist_band_n[band]
                        r_valid_band[band] = False
                        if slot < topk:
                            row = staged_indices[hoist_band_n[band], loop_iter & Int32(1)]
                            r_valid_band[band] = row >= Int32(0) and row < mKV.shape[0]
                assert cute.size(r_score) == self.N_TILE_CTA
                for h_group in cutlass.range_constexpr(4):
                    lse = hoist_lse[h_group]
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        v0, v1 = cute.arch.fma_packed_f32x2((r_score[i0], r_score[i1]), (softmax_scale_log2_e, softmax_scale_log2_e), (lse, lse))
                        v0 = cute.arch.fmin(v0, Float32(64.0), nan=True)
                        v1 = cute.arch.fmin(v1, Float32(64.0), nan=True)
                        v0 = cute.math.exp2(v0, fastmath=True)
                        v1 = cute.math.exp2(v1, fastmath=True)
                        # Zero-filled K/V cannot mask an overflowing dS product.
                        if not r_valid_band[2 * pair]:
                            v0 = Float32(0.0)
                        if not r_valid_band[2 * pair + 1]:
                            v1 = Float32(0.0)
                        r_score[i0] = v0
                        r_score[i1] = v1
                        if cutlass.const_expr(self.deterministic):
                            r_p_words[i0 // 2] = _dq_pack_bf16x2(v0, v1)
                        else:
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
                if cutlass.const_expr(self.deterministic and self.USE_P_LOW):
                    for h_group in cutlass.range_constexpr(4):
                        for pair in cutlass.range_constexpr(4):
                            i0 = hoist_band_indices[h_group][2 * pair]
                            i1 = hoist_band_indices[h_group][2 * pair + 1]
                            high0, high1 = _bf16x2_to_f32x2(r_p_words[i0 // 2])
                            r_p_words[i0 // 2] = _dq_pack_bf16x2(_finite_residual_f32(r_score[i0], high0), _finite_residual_f32(r_score[i1], high1))
                    r_p_low_store = thread_copy_r2s.retile(r_p)
                    if math_owns_local_half:
                        cute.copy(tiled_copy_r2s, r_p_low_store, t_rs_ds_local_tile)
                    else:
                        if cutlass.const_expr(plo_staging):
                            if loop_iter > Int32(0):
                                # The previous tile's P_lo staging has landed in the peer (ds_stage_free).
                                _mbarrier_wait_acquire_cluster(ds_stage_free_mbar, (loop_iter - Int32(1)) & Int32(1))
                        cute.copy(tiled_copy_r2s, r_p_low_store, t_rs_p_lo_xchg_tile)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(pds_ready_mbars)
                    pds_state.advance()
                pipe_dp_done.consumer_wait(dp_state)
                if dp_state.index == Int32(0):
                    cute.copy(dp_copy, dp_source, r_dp)
                else:
                    cute.copy(dp_copy_pp, dp_source_pp, r_dp)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.deterministic):
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipe_dp_done.consumer_release(dp_state)
                else:
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
                        d0, d1 = cute.arch.mul_packed_f32x2((d0, d1), (scale_softmax, scale_softmax))
                        if not r_valid_band[2 * pair]:
                            d0 = Float32(0.0)
                        if not r_valid_band[2 * pair + 1]:
                            d1 = Float32(0.0)
                        if cutlass.const_expr(self.deterministic and self.USE_DS_LOW):
                            r_dp[i0] = d0
                            r_dp[i1] = d1
                        if cutlass.const_expr(self.deterministic):
                            r_p_words[i0 // 2] = _dq_pack_bf16x2(d0, d1)
                        else:
                            r_ds[i0] = self.element_dtype(d0)
                            r_ds[i1] = self.element_dtype(d1)
                r_ds_store = thread_copy_r2s.retile(r_ds)
                assert t_rs_ds_local_tile.shape == r_ds_store.shape
                assert t_rs_ds_tile.shape == r_ds_store.shape
                if cutlass.const_expr(self.deterministic):
                    # The dQ operand (ds_image) goes out first: it only waits until the peer holds this
                    # CTA's P_lo staging (none with ``plo_staging``), not for the hi/lo sweeps that still
                    # read the dS blocks.
                    if cutlass.const_expr(not plo_staging):
                        _mbarrier_wait_acquire_cluster(ds_stage_free_mbar, loop_iter & Int32(1))
                    cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_tile)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if cutlass.const_expr(use_ds2):
                        # The dQ products wait only for the dS images, not for the relay lane's P_lo hand-off.
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(ds_local_ready_mbar, Int32(0))
                    else:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(dsq_ready_mbar)
                    if cutlass.const_expr(use_ds2):
                        # Dedicated dS blocks: only the previous tile's dK products must be done.
                        pipe_ds2.producer_acquire(ds2_state)
                        if math_owns_local_half:
                            cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds2_local_tile)
                    else:
                        pipe_pds.producer_acquire(pds_state)
                        if math_owns_local_half:
                            cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_local_tile)
                else:
                    if math_owns_local_half:
                        cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_local_tile)
                    cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_tile)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(pds_ready_mbars)
                if cutlass.const_expr(self.deterministic and self.USE_DS_LOW):
                    for h_group in cutlass.range_constexpr(4):
                        for pair in cutlass.range_constexpr(4):
                            i0 = hoist_band_indices[h_group][2 * pair]
                            i1 = hoist_band_indices[h_group][2 * pair + 1]
                            r_ds[i0] = self.element_dtype(_finite_residual_f32(r_dp[i0], Float32(r_ds[i0])))
                            r_ds[i1] = self.element_dtype(_finite_residual_f32(r_dp[i1], Float32(r_ds[i1])))
                    r_ds_low_store = thread_copy_r2s.retile(r_ds)
                    if math_owns_local_half:
                        cute.copy(tiled_copy_r2s, r_ds_low_store, t_rs_p_local_tile)
                    else:
                        cute.copy(tiled_copy_r2s, r_ds_low_store, t_rs_p_xchg_tile)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(p_ready_mbars)
                if cutlass.const_expr(use_ds2):
                    ds2_state.advance()
                else:
                    pds_state.advance()
            if cutlass.const_expr(plo_staging and self.USE_P_LOW):
                if tile_count > Int32(0):
                    if not math_owns_local_half:
                        # The peer's arrive for the last P_lo staging must land before this CTA exits.
                        _mbarrier_wait_acquire_cluster(ds_stage_free_mbar, (tile_count - Int32(1)) & Int32(1))
            if tile_count > Int32(0):
                pipe_dq_done.consumer_wait(dq_done_state)
                if cutlass.const_expr(self.deterministic):
                    # DQ_EPI_MATH_BATCH-chunk batches fit the math register budget, so this round does not wait
                    # for the reducers to retire their registers after the last drain.
                    self._store_dq_epi_scalar_direct(
                        t_dq[1], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, mdQ, 1, token_idx, batch_idx, rank, mtx, self.DQ_EPI_MATH_BATCH
                    )
                else:
                    cute.arch.setmaxregister_increase(self.MATH_EPI_SETMAXREG)
                    self._store_dq_epi_scalar_direct(
                        t_dq[1], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, mdQ, 1, token_idx, batch_idx, rank, mtx, self.DQ_EPI_BATCH_CHUNKS
                    )
                pipe_dq_done.consumer_release(dq_done_state)
                dq_done_state.advance()
            else:
                self._zero_dq(rank_dq_coordinates, mdQ, 0, token_idx, batch_idx, mtx)
                self._zero_dq(rank_dq_coordinates, mdQ, 1, token_idx, batch_idx, mtx)
        elif warp_idx < Int32(self.MMA_WARP):
            if cutlass.const_expr(self.deterministic):
                self._set_role_registers(self.REDUCER_SETMAXREG)
            rtx = tidx - Int32(self.REDUCE_THREAD_BEGIN)
            if cutlass.const_expr(self.max_topk <= self.O_PREFETCH_MAX_TOPK):
                self._prefetch_odo_rows(mOut, token_idx, batch_idx, rank, rtx)
            self._compute_folded_lse(mLSE, mAttnSink, softmax_stats, token_idx, batch_idx, rank, rtx)
            cute.arch.fence_view_async_shared()
            if tile_count > Int32(0):
                self.stats_lse_barrier.arrive_unaligned()
            if tile_count > Int32(0):
                self._compute_stationary_odo(
                    mOut, mdO, stationary_do_physical, stationary_do_rows, stationary_tma_mbars, softmax_stats, token_idx, batch_idx, rank, rtx, scale_softmax
                )
            else:
                self._compute_global_odo(mOut, mdO, softmax_stats, token_idx, batch_idx, rank, rtx, scale_softmax)
            cute.arch.fence_view_async_shared()
            if tile_count > Int32(0):
                self.stats_odo_barrier.arrive_unaligned()
            self.dsink_reducer_barrier.arrive_and_wait()
            self._publish_dsink_stats(sum_odo, scaled_lse, softmax_stats, token_idx, batch_idx, rank, rtx)
            dkv_wait = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            dkv_rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            index_row = mTopkIdxs[None, (token_idx, batch_idx)]
            if cutlass.const_expr(self.deterministic and self.rank_shard):
                # Destination codes and dKV records belong to this wave's setup launch, which
                # also waited for the previous main kernel; then release the next setup.
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
                rank_pitch, rank_planes = self.rank_geometry(mKV.shape[0])
                drain_offsets_base = None
                drain_record_base = None
                if cutlass.const_expr(self.RANK_DRAIN_OVERLAP_LOOKUPS):
                    drain_offsets_base = self.rank_offsets_parity_base(mdKV_acc, rank_pitch, rank_planes, rank_parity)
                    drain_record_base = rank_planes * Int32(rank_pitch) + self.rank_record_base(rank_parity)
                if cutlass.const_expr(self.rank_fused):
                    tail_grid_x, _, _ = cute.arch.grid_dim()
                    self._rank_tail(
                        mdKV_acc,
                        mTopkIdxs,
                        mTopkLength,
                        mKV.shape[0],
                        tail_step,
                        tail_begin,
                        tail_size,
                        physical_x * Int32(self.REDUCE_WARPS) + (warp_idx - Int32(self.REDUCE_WARP_BEGIN)),
                        tail_grid_x * Int32(self.REDUCE_WARPS),
                        rtx % Int32(32),
                        query_stride,
                        Int32(0),
                    )
            for loop_iter in cutlass.range(tile_count):
                tile_index = self._visit_tile(loop_iter, tile_count, sweep_ascending)
                if cutlass.const_expr(self.deterministic and self.rank_shard):
                    dkv_wait, dkv_rel = self._drain_dkv_rank(
                        t_dkv[0],
                        t_dkv[1],
                        mdKV_acc,
                        index_row,
                        tile_index,
                        topk,
                        mKV.shape[0],
                        rank_pitch,
                        rank_planes,
                        rank_parity,
                        query_shard,
                        rtx,
                        rank,
                        pipe_dkv_done,
                        dkv_wait,
                        dkv_rel,
                        drain_offsets_base,
                        drain_record_base,
                        loop_iter % Int32(2) if self.ROUND_REUSE else Int32(0),
                    )
                elif cutlass.const_expr(self.deterministic):
                    if cutlass.const_expr(self.split_wg_queries_per_wave > 0):
                        dkv_wait, dkv_rel = self._drain_dkv_deterministic(
                            t_dkv[0], t_dkv[1], mdKV_acc, index_row, tile_index, topk, batch_idx, rtx, rank, split_wg_shard, pipe_dkv_done, dkv_wait, dkv_rel
                        )
                    else:
                        dkv_wait, dkv_rel = self._drain_dkv(
                            t_dkv[0], t_dkv[1], mdKV_acc, index_row, tile_index, topk, batch_idx, rtx, rank, pipe_dkv_done, dkv_wait, dkv_rel
                        )
                else:
                    dkv_wait, dkv_rel = self._drain_dkv(
                        t_dkv[0], t_dkv[1], mdKV_acc, index_row, tile_index, topk, batch_idx, rtx, rank, pipe_dkv_done, dkv_wait, dkv_rel
                    )
            cute.arch.setmaxregister_decrease(self.REDUCER_EPI_SETMAXREG)
        else:
            # Warps 16-19 form one warpgroup: one warpgroup-uniform budget before the role split.
            if cutlass.const_expr(self.deterministic):
                self._set_role_registers(self.UTILITY_SETMAXREG)
            if warp_idx == Int32(self.MMA_WARP):
                if is_leader_cta:
                    s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                    dp_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                    kscore_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    round_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ROUND_STAGES)
                    pds_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    ds2_cons = None
                    if cutlass.const_expr(use_ds2):
                        ds2_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    dkv_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                    dkv_com = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                    dq_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                    dq0_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                    if tile_count > Int32(0):
                        _mbarrier_wait_acquire_cluster(stationary_ready_mbar, Int32(0))
                        pipe_dq0_done.producer_acquire(dq0_done_prod)
                    pipe_dq_done.producer_acquire(dq_done_prod)
                    relay_phase = Int32(0)
                    for loop_iter in cutlass.range(tile_count):
                        pipe_kscore.consumer_wait(kscore_cons)
                        s_prod = self._issue_score(score_tiled_mma, t_score, t_score_pp, score_q_fragment, score_k_fragment, pipe_s_done, s_prod)
                        if loop_iter == Int32(0):
                            _mbarrier_wait_acquire_cluster(stationary_ready_mbar + 1, Int32(0))
                        dp_prod = self._issue_score(dp_tiled_mma, t_dp, t_dp_pp, score_do_fragment, dp_k_fragment, pipe_dp_done, dp_prod)
                        if cutlass.const_expr(self.DV0_TMEM_A):
                            if loop_iter == Int32(0):
                                _mbarrier_wait_acquire_cluster(storage.dot_tmem_ready_mbar.data_ptr(), Int32(0))
                                _tcgen05_fence_after_thread_sync()
                        pipe_kscore.consumer_release(kscore_cons)
                        kscore_cons.advance()
                        dq_acc = loop_iter != Int32(0)
                        if cutlass.const_expr(use_ds2):
                            round_cons, kscore_cons, dkv_acq, dkv_com, pds_cons, ds2_cons = self._issue_grads_early_dq_hilo(
                                dq_tiled_mma,
                                dkv_tiled_mma,
                                dkv_ts_tiled_mma,
                                dot_tmem_fragment,
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
                                None,
                                pipe_ds2,
                                ds2_cons,
                                ds2_fragments[0],
                                ds2_fragments[1],
                                dot_tmem_fragment_1,
                                loop_iter,
                                tile_count,
                            )
                        elif cutlass.const_expr(self.deterministic):
                            round_cons, kscore_cons, dkv_acq, dkv_com, pds_cons = self._issue_grads_early_dq_hilo(
                                dq_tiled_mma,
                                dkv_tiled_mma,
                                dkv_ts_tiled_mma,
                                dot_tmem_fragment,
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
                                rolled_fragments,
                            )
                        else:
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
                        if cutlass.const_expr(use_ds2):
                            # The dS blocks are free once this tile's dK products are done.
                            pipe_ds2.consumer_release(ds2_cons)
                            ds2_cons.advance()
                        else:
                            pipe_pds.consumer_release(pds_cons)
                            pds_cons.advance()
                        relay_phase = Int32(1) - relay_phase
                    if tile_count > Int32(0):
                        dq0_done_prod.advance()
                        dq_done_prod.advance()
                        pipe_s_done.producer_tail(s_prod)
                        pipe_dp_done.producer_tail(dp_prod)
                        pipe_dkv_done.producer_tail(dkv_com)
                        pipe_dq_done.producer_tail(dq_done_prod)
                        pipe_dq0_done.producer_tail(dq0_done_prod)
            elif warp_idx == Int32(self.LOAD_WARP):
                if cutlass.const_expr(early_stationary):
                    if tile_count == Int32(0):
                        # The pre-rendezvous Q/dO loads still land in this CTA's shared memory before it exits.
                        cute.arch.mbarrier_wait(stationary_tma_mbars, Int32(0))
                        cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
                if tile_count > Int32(0):
                    if cutlass.const_expr(not early_stationary):
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
                    for loop_iter in cutlass.range(tile_count):
                        gens_per_group = self.H_TILE_CLUSTER // self.ROUND_K_HEADS
                        if cutlass.const_expr(self.deterministic):
                            # With ``DV1_TMEM_A`` (which ``DV0_TMEM_A`` plans always set) both dO^T groups stay in
                            # TMEM: the ring streams the two Q^T groups.
                            skipped_gens = 2 * gens_per_group if cutlass.const_expr(self.DV1_TMEM_A) else 0
                            ring_gens = self.ROUND_GENS_PER_TILE - skipped_gens
                            for micro_gen in cutlass.range_constexpr(skipped_gens, self.ROUND_GENS_PER_TILE):
                                tensor_kind = micro_gen // (2 * gens_per_group)
                                grad_round = micro_gen // gens_per_group % 2
                                h_half = micro_gen % gens_per_group
                                ring_gen = micro_gen - skipped_gens
                                round_slot = ring_gen % self.ROUND_STAGES
                                round_count = loop_iter * Int32(ring_gens) + Int32(ring_gen)
                                load_group = cutlass.Boolean(True)
                                if cutlass.const_expr(self.ROUND_REUSE):
                                    # Tile t sweeps D-rounds (t % 2, 1 - t % 2); after tile 0 its first Q^T group is
                                    # still in the ring, so only the second loads (4 generations per group, one group
                                    # block per tile plus tile 0's extra one).
                                    group_index = ring_gen // gens_per_group
                                    grad_round = (Int32(group_index) + loop_iter) % Int32(2)
                                    round_count = Int32(gens_per_group) * (loop_iter + Int32(group_index)) + Int32(h_half)
                                    if loop_iter > Int32(0) and group_index == 0:
                                        load_group = cutlass.Boolean(False)
                                if load_group:
                                    round_acq = pipeline.PipelineState(
                                        self.ROUND_STAGES,
                                        round_count,
                                        Int32(round_slot),
                                        Int32(1) ^ (round_count // Int32(self.ROUND_STAGES) & Int32(1)),
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
                        else:
                            for micro_gen in cutlass.range_constexpr(self.ROUND_GENS_PER_TILE):
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
                    if cutlass.const_expr(self.deterministic):
                        tail_gens = self.ROUND_GENS_PER_TILE - (2 * (self.H_TILE_CLUSTER // self.ROUND_K_HEADS) if cutlass.const_expr(self.DV1_TMEM_A) else 0)
                        tail_count = tile_count * Int32(tail_gens)
                        if cutlass.const_expr(self.ROUND_REUSE):
                            # One Q^T group block per tile plus tile 0's first group.
                            tail_count = Int32(self.H_TILE_CLUSTER // self.ROUND_K_HEADS) * (tile_count + min(tile_count, Int32(1)))
                        round_tail = pipeline.PipelineState(
                            self.ROUND_STAGES, tail_count, tail_count % Int32(self.ROUND_STAGES), Int32(1) ^ (tail_count // Int32(self.ROUND_STAGES) & Int32(1))
                        )
                    else:
                        round_tail = pipeline.PipelineState(self.ROUND_STAGES, tile_count * Int32(self.ROUND_GENS_PER_TILE), Int32(0), Int32(1))
                    pipe_round.producer_tail(round_tail)
            elif warp_idx == Int32(self.RELAY_WARP):
                relay_lane = tidx % Int32(32)
                if relay_lane == Int32(0):
                    if cutlass.const_expr(use_ds2):
                        for loop_iter in cutlass.range(tile_count):
                            p_phase = loop_iter & Int32(1)
                            cute.arch.mbarrier_wait(p_ready_mbars, p_phase)
                            cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                            if rank == Int32(0):
                                _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[0], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                            else:
                                _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[1], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                            # The high dV products need only P_hi: report its landing before waiting for P_lo.
                            _mbarrier_wait_acquire_cluster(landing_mbars, p_phase)
                            cute.arch.mbarrier_arrive(relay_mbars, Int32(0))
                            cute.arch.mbarrier_wait(pds_ready_mbars, Int32(0))
                            cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                            if rank == Int32(0):
                                _cpasync_bulk_s2cluster(ds_image_raw + Int32(2048), ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            else:
                                _cpasync_bulk_s2cluster(ds_image_raw, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            _mbarrier_wait_acquire_cluster(landing_mbars + 1, Int32(0))
                            # The peer's P_hi and P_lo staging have landed here: its ds_image may now take dS.
                            cute.arch.mbarrier_arrive(ds_stage_free_mbar, peer_rank)
                            p_commit = pipeline.PipelineState(1, loop_iter, Int32(0), Int32(1) ^ loop_iter & Int32(1))
                            pipe_pds.producer_commit(p_commit)
                            cute.arch.mbarrier_wait(pds_ready_mbars, Int32(1))
                            cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                            if rank == Int32(0):
                                _cpasync_bulk_s2cluster(ds_image_raw + Int32(2048), ds2_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            else:
                                _cpasync_bulk_s2cluster(ds_image_raw, ds2_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            ds2_commit = pipeline.PipelineState(1, loop_iter, Int32(0), Int32(1) ^ loop_iter & Int32(1))
                            pipe_ds2.producer_commit(ds2_commit)
                            _mbarrier_wait_acquire_cluster(landing_mbars + 1, Int32(1))
                            cute.arch.mbarrier_arrive(relay_mbars + 1, Int32(0))
                        if tile_count > Int32(0):
                            pds_tail = pipeline.PipelineState(1, tile_count, Int32(0), Int32(1) ^ tile_count & Int32(1))
                            pipe_pds.producer_tail(pds_tail)
                            ds2_tail = pipeline.PipelineState(1, tile_count, Int32(0), Int32(1) ^ tile_count & Int32(1))
                            pipe_ds2.producer_tail(ds2_tail)
                    elif cutlass.const_expr(self.deterministic):
                        for loop_iter in cutlass.range(tile_count):
                            p_phase = loop_iter & Int32(1)
                            cute.arch.mbarrier_wait(p_ready_mbars, p_phase)
                            cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                            if rank == Int32(0):
                                _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[0], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                            else:
                                _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[1], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                            cute.arch.mbarrier_wait(pds_ready_mbars, Int32(0))
                            cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                            if cutlass.const_expr(plo_staging):
                                if rank == Int32(0):
                                    _cpasync_bulk_s2cluster(p_lo_xchg_ptr, ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                                else:
                                    _cpasync_bulk_s2cluster(p_lo_xchg_ptr, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            else:
                                if rank == Int32(0):
                                    _cpasync_bulk_s2cluster(
                                        ds_image_raw + Int32(2048), ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank
                                    )
                                else:
                                    _cpasync_bulk_s2cluster(ds_image_raw, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            _mbarrier_wait_acquire_cluster(landing_mbars, p_phase)
                            cute.arch.mbarrier_arrive(relay_mbars, Int32(0))
                            _mbarrier_wait_acquire_cluster(landing_mbars + 1, Int32(0))
                            # The peer's P_lo staging has landed here: its ds_image may now take dS.
                            cute.arch.mbarrier_arrive(ds_stage_free_mbar, peer_rank)
                            p_generation = loop_iter * Int32(2)
                            p_commit = pipeline.PipelineState(1, p_generation, Int32(0), Int32(1) ^ p_generation & Int32(1))
                            pipe_pds.producer_commit(p_commit)
                            cute.arch.mbarrier_wait(dsq_ready_mbar, p_phase)
                            cute.arch.mbarrier_arrive(ds_local_ready_mbar, Int32(0))
                            ds_generation = p_generation + Int32(1)
                            if cutlass.const_expr(self.DS_RELAY_ON_RELEASE):
                                # The peer's dS block may land once the sweeps released the P_lo blocks of both
                                # CTAs (the dS image is already in place): send it without waiting for the local
                                # dS-block store, which the commit below still waits for.
                                pipe_pds.producer_acquire(pipeline.PipelineState(1, ds_generation, Int32(0), Int32(1) ^ ds_generation & Int32(1)))
                            else:
                                cute.arch.mbarrier_wait(pds_ready_mbars, Int32(1))
                            cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                            if rank == Int32(0):
                                _cpasync_bulk_s2cluster(ds_image_raw + Int32(2048), ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            else:
                                _cpasync_bulk_s2cluster(ds_image_raw, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                            if cutlass.const_expr(self.DS_RELAY_ON_RELEASE):
                                cute.arch.mbarrier_wait(pds_ready_mbars, Int32(1))
                            if cutlass.const_expr(self.USE_DS_LOW):
                                cute.arch.mbarrier_wait(p_ready_mbars, Int32(1))
                                cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                                if rank == Int32(0):
                                    _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[0], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                                else:
                                    _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[1], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                                _mbarrier_wait_acquire_cluster(landing_mbars, Int32(1))
                            ds_commit = pipeline.PipelineState(1, ds_generation, Int32(0), Int32(1) ^ ds_generation & Int32(1))
                            pipe_pds.producer_commit(ds_commit)
                            _mbarrier_wait_acquire_cluster(landing_mbars + 1, Int32(1))
                            cute.arch.mbarrier_arrive(relay_mbars + 1, Int32(0))
                        if tile_count > Int32(0):
                            tail_generation = tile_count * Int32(2)
                            pds_tail = pipeline.PipelineState(1, tail_generation, Int32(0), Int32(1) ^ tail_generation & Int32(1))
                            pipe_pds.producer_tail(pds_tail)
                    else:
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
            elif warp_idx == Int32(self.SETUP_WARP):
                if cutlass.const_expr(self.deterministic and self.rank_shard and self.rank_fused):
                    # These phases read the row state, plane-use and mark entries that the preceding launch
                    # wrote (the previous main kernel's setup warp, or the leading setup launches).
                    cute.arch.griddepcontrol_wait()
                    setup_grid_x, _, _ = cute.arch.grid_dim()
                    self._rank_setup_slots(
                        mdKV_acc,
                        mTopkIdxs,
                        mTopkLength,
                        mKV.shape[0],
                        setup_step,
                        setup_prev_begin,
                        setup_prev_size,
                        setup_cur_begin,
                        setup_cur_size,
                        setup_next_begin,
                        setup_next_size,
                        physical_x * Int32(32) + tidx % Int32(32),
                        setup_grid_x * Int32(32),
                        query_stride,
                    )
        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        if cutlass.const_expr(self.deterministic):
            # The exit rendezvous only guards shared and tensor memory lifetime: every remote mbarrier arrive,
            # DSMEM copy and paired MMA that touches the peer's storage was already awaited by its receiver,
            # so the arrive need not first drain this thread's outstanding global stores (relaxed arrive).
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp_idx == Int32(self.MATH_WARP_BEGIN):
            cute.arch.dealloc_tmem(tmem_ptr, self.TMEM_COLUMNS, is_two_cta=True)

    @staticmethod
    def _make_umma_async_pipeline(num_stages, producer_group, consumer_group, barrier_storage, cluster_layout_vmnk):
        """UMMA-producer pipeline whose consumer releases target cluster rank 0.

        For thread arrivals the mask is the destination CTA rank: the consumers of both
        CTAs release on the leader CTA's (rank 0) empty barrier, which its next paired MMA
        acquires. The trace-time 0 replaces the leader rank ``create`` derives at run time.
        The leader's commits keep the multicast mask that signals both CTAs.
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
        """UMMA-consumer pipeline whose producer commits target cluster rank 0.

        For thread arrivals the mask is the destination CTA rank: the producers of both
        CTAs commit to the leader CTA's (rank 0) full barrier, on which its paired MMA
        waits. The trace-time 0 replaces the leader rank ``create`` derives at run time.
        The leader's releases keep the multicast mask that signals both CTAs.
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
        self, score_a_layout_staged, score_b_layout_staged, dkv_a_layout_staged, dkv_b_layout_staged, dq_a_layout_staged, dq_b_layout_staged
    ):
        """Lay out the mbarrier arrays and the staged operand buffers."""
        element_dtype = self.element_dtype
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 8192
        assert cute.cosize(score_b_layout_staged) == 2 * cute.cosize(dkv_a_layout_staged)
        assert cute.cosize(dkv_a_layout_staged) == 8192
        assert cute.cosize(dkv_b_layout_staged) <= 2048
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096
        deterministic = self.deterministic
        plo_elements = self.PDS_BLOCK_ELEMENTS if self.deterministic and self.max_topk <= self.PLO_STAGING_MAX_TOPK else 8

        if self._use_ds2():
            return self._make_shared_storage_ds2(element_dtype)

        # Fields under ``if deterministic`` exist only in the deterministic layouts (a class body records an
        # annotation only when it executes), so the non-deterministic layout keeps its own field list.

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
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            p_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            ds_local_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            if deterministic:
                ds_stage_free_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                dsq_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                stationary_free_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                dot_tmem_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
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
            if deterministic:
                # Peer-half P_lo staging (see ``PLO_STAGING_MAX_TOPK``); a stub otherwise.
                p_lo_xchg: cute.struct.Align[cute.struct.MemRange[element_dtype, plo_elements], 1024]

        assert SharedStorage.size_in_bytes() <= self.MAX_SMEM_BYTES
        return SharedStorage

    def _make_shared_storage_ds2(self, element_dtype):
        """``_make_shared_storage`` with dedicated dS blocks (``ds2_blocks``) and their pipeline barriers, without ``p_xchg``."""

        @cute.struct
        class SharedStorage:
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            kscore_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 16]
            pds_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            ds2_mbars: cute.struct.MemRange[cutlass.Int64, 2]
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
            ds_stage_free_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            dsq_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            stationary_free_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            dot_tmem_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            stationary_q: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            stationary_do: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            score_kv: cute.struct.Align[cute.struct.MemRange[element_dtype, 16384], 1024]
            round_buf: cute.struct.Align[cute.struct.MemRange[element_dtype, 16384], 1024]
            p_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            ds_image: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            ds_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            ds2_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
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
                    cute.gemm(mma, t_dq_0, kd_fragment_a[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_0)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                if commit_final:
                    cute.arch.fence_view_async_tmem_store()
                    dq0_done_pipeline.producer_commit(dq0_done_state)
            else:
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_b, mode=[2])):
                    cute.gemm(mma, t_dq_1, kd_fragment_b[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_1)
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
        return (round_consumer_state, kscore_consumer_state, dkv_acquire_state, dkv_commit_state, pds_consumer_state)


class FlashAttentionDSABackwardSm100H128TwoCTADeterministic(TwoCTADeterministicReduction, FlashAttentionDSABackwardSm100H128TwoCTA):
    """The H128 D512 two-CTA kernel with ordered scratch scatter and reduction."""

    FUSED_FOLD_MIN_ROWS = 1024
    USE_P_LOW = True
    USE_DS_LOW = False
    # The reducers do not prefetch O ahead of the LSE fold.
    O_PREFETCH_MAX_TOPK = 0

    def __init__(self, element_dtype: Type[cutlass.Numeric], head_dim: int, head_dim_v: int, block_tile: int, max_topk: int = 0):
        super().__init__(element_dtype, head_dim, head_dim_v, block_tile, max_topk)
        # Larger top-k capacities give the first queries separate warp-group
        # planes. The remaining queries share a plane and serialize its writers.
        if max_topk >= 1024:
            self.num_dkv_shards = 112
            self.split_wg_queries_per_wave = 38
        else:
            self.num_dkv_shards = 74
            self.split_wg_queries_per_wave = 0

    @cute.jit
    def postprocess_deterministic(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        active_shards: Int32,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        problem_shape,
        stream: cuda.CUstream,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Hide ordered dSink work in the terminal dKV folds when they have two stages."""
        active_groups = cute.ceil_div(active_shards, self.dkv_fold_group_size)
        if seqlen >= Int32(self.FUSED_FOLD_MIN_ROWS):
            self.sum_dSink_partials_deterministic(sum_odo, scaled_lse, attn_sink, problem_shape, chunk_size).launch(
                grid=[cute.ceil_div(problem_shape[0], chunk_size), 1, 1], block=[128, 1, 1], stream=stream
            )
            self.fold_convert_dkv_finalize_dSink_deterministic(mdKV_acc, mdKV, seqlen, active_shards, sum_odo, d_sink, problem_shape, chunk_size).launch(
                grid=[cute.ceil_div(seqlen, 4), 1, 1], block=[32, 4, 1], stream=stream
            )
        else:
            self.fold_dKV_shards_dSink_partials(mdKV_acc, seqlen, active_shards, sum_odo, scaled_lse, attn_sink, problem_shape, chunk_size).launch(
                grid=[seqlen, active_groups, 1], block=[128, 1, 1], stream=stream
            )
            self.fold_groups_convert_dkv_finalize_dSink_deterministic(mdKV_acc, mdKV, seqlen, active_groups, sum_odo, d_sink, problem_shape, chunk_size).launch(
                grid=[seqlen, 1, 1], block=[32, 1, 1], stream=stream
            )

    @cute.jit
    def postprocess_rank(
        self,
        mRows: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        last_step: Int32,
        last_begin: Int32,
        last_size: Int32,
        last_stride: Int32,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        d_sink: cute.Tensor,
        problem_shape,
        stream: cuda.CUstream,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Finish the last wave with the ordered dSink partials, then fold each row's used planes and finalize dSink."""
        self.launch_rank_final(
            mRows,
            mTopkIdxs,
            mTopkLength,
            seqlen,
            last_step,
            last_begin,
            last_size,
            last_stride,
            sum_odo,
            scaled_lse,
            attn_sink,
            problem_shape,
            stream,
            chunk_size,
        )
        self.launch_rank_fold(mRows, mdKV, seqlen, sum_odo, d_sink, problem_shape, stream, chunk_size)

    @cute.kernel
    def zero_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32):
        """Initialize D512 scratch with aligned 128-bit positive-zero stores."""
        block_row, shard, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        assert self.head_dim == 512
        pitch = cute.round_up(seqlen, 8)
        dim_base = tidx * Int32(4)
        for row_offset in cutlass.range_constexpr(4):
            row = block_row * Int32(4) + Int32(row_offset)
            if row < seqlen:
                acc_row = mdKV_acc[None, row + shard * pitch, (0, 0)]
                _store_f32x4_zero(acc_row.iterator + dim_base)

    @cute.kernel
    def fold_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32, active_shards: Int32):
        """Fold each ordered eight-shard group four FP32 columns per thread."""
        row, group, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        assert self.head_dim == 512
        pitch = cute.round_up(seqlen, 8)
        group_begin = group * Int32(self.dkv_fold_group_size)
        dim_base = tidx * Int32(4)
        if row < seqlen:
            acc_0 = Float32(0.0)
            acc_1 = Float32(0.0)
            acc_2 = Float32(0.0)
            acc_3 = Float32(0.0)
            for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                shard = group_begin + Int32(local_shard)
                if shard < active_shards:
                    acc_row = mdKV_acc[None, row + shard * pitch, (0, 0)]
                    values = _load_f32x4(acc_row.iterator + dim_base)
                    acc_0 += values[0]
                    acc_1 += values[1]
                    acc_2 += values[2]
                    acc_3 += values[3]
            out_row = mdKV_acc[None, row + group_begin * pitch, (0, 0)]
            _store_f32x4(out_row.iterator + dim_base, acc_0, acc_1, acc_2, acc_3)

    @cute.kernel
    def fold_dKV_shards_dSink_partials(
        self,
        mdKV_acc: cute.Tensor,
        seqlen: Int32,
        active_shards: Int32,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        problem_shape,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Fold shard groups while group-zero CTAs form ordered dSink partials."""
        row, group, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        assert self.head_dim == 512
        pitch = cute.round_up(seqlen, 8)
        group_begin = group * Int32(self.dkv_fold_group_size)
        dim_base = tidx * Int32(4)
        if row < seqlen:
            acc_0 = Float32(0.0)
            acc_1 = Float32(0.0)
            acc_2 = Float32(0.0)
            acc_3 = Float32(0.0)
            for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                shard = group_begin + Int32(local_shard)
                if shard < active_shards:
                    acc_row = mdKV_acc[None, row + shard * pitch, (0, 0)]
                    values = _load_f32x4(acc_row.iterator + dim_base)
                    acc_0 += values[0]
                    acc_1 += values[1]
                    acc_2 += values[2]
                    acc_3 += values[3]
            out_row = mdKV_acc[None, row + group_begin * pitch, (0, 0)]
            _store_f32x4(out_row.iterator + dim_base, acc_0, acc_1, acc_2, acc_3)
        if group == Int32(0):
            chunk_count = cute.ceil_div(problem_shape[0], chunk_size)
            q_block_idx = row
            while q_block_idx < chunk_count:
                self._sum_dSink_partial_head_deterministic(sum_odo, scaled_lse, attn_sink, problem_shape, q_block_idx, tidx, chunk_size)
                q_block_idx += seqlen

    @cute.kernel
    def fold_dKV_groups(self, mdKV_acc: cute.Tensor, seqlen: Int32, active_groups: Int32):
        """Fold ordered group partials four FP32 columns per thread."""
        row, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        assert self.head_dim == 512
        pitch = cute.round_up(seqlen, 8)
        dim_base = tidx * Int32(4)
        if row < seqlen:
            acc_0 = Float32(0.0)
            acc_1 = Float32(0.0)
            acc_2 = Float32(0.0)
            acc_3 = Float32(0.0)
            for group in cutlass.range(active_groups):
                acc_row = mdKV_acc[None, row + group * Int32(self.dkv_fold_group_size) * pitch, (0, 0)]
                values = _load_f32x4(acc_row.iterator + dim_base)
                acc_0 += values[0]
                acc_1 += values[1]
                acc_2 += values[2]
                acc_3 += values[3]
            out_row = mdKV_acc[None, row, (0, 0)]
            _store_f32x4(out_row.iterator + dim_base, acc_0, acc_1, acc_2, acc_3)

    @cute.jit
    def fold_dKV_accumulator(self, mdKV_acc: cute.Tensor, seqlen: Int32, stream: cuda.CUstream, active_shards: Int32 = 128):
        """Launch D512 vector folds while preserving the fixed reduction tree."""
        active_groups = cute.ceil_div(active_shards, self.dkv_fold_group_size)
        self.fold_dKV_shards(mdKV_acc, seqlen, active_shards).launch(grid=[seqlen, active_groups, 1], block=[128, 1, 1], stream=stream)
        self.fold_dKV_groups(mdKV_acc, seqlen, active_groups).launch(grid=[seqlen, 1, 1], block=[128, 1, 1], stream=stream)
        return cute.make_tensor(mdKV_acc.iterator, cute.make_layout((self.head_dim, seqlen, (1, 1)), stride=(1, Int64(self.head_dim), (0, 0))))

    @cute.jit
    def _fill_dot_tmem(
        self, t_dot_fill: cute.Tensor, mdO: cute.Tensor, token_idx: Int32, batch_idx: Int32, rank: Int32, mtx: Int32, d_round_base: cutlass.Constexpr[int] = 0
    ):
        """Store this CTA's D-round-0 dO^T rows into TMEM: lane = D row of the CTA's 128-row slice, column = head pair.

        Column ``j`` packs heads ``2 j`` (low half) and ``2 j + 1`` (high half), the K-major BF16 layout of an
        A operand read from tensor memory. The caller orders the stores before the MMA with a barrier.
        """
        t_fill_core = t_dot_fill[(None, None), 0, 0]
        store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_r2t = tcgen05.make_tmem_copy(store_atom, t_fill_core)
        thread_r2t = tiled_r2t.get_slice(mtx)
        fill_destination = thread_r2t.partition_D(t_fill_core)
        fill_coordinates = thread_r2t.partition_S(cute.make_identity_tensor((self.D_TILE_CTA, self.N_TILE)))
        fill_values = cute.make_rmem_tensor(fill_coordinates.shape, self.acc_dtype)
        fill_words = cute.recast_tensor(fill_values, cutlass.Uint32)
        d_base = rank * Int32(self.D_TILE_CTA) + Int32(d_round_base)
        for i in cutlass.range_constexpr(cute.size(fill_values)):
            d = d_base + Int32(cute.get(fill_coordinates[i], mode=[0]))
            head = Int32(cute.get(fill_coordinates[i], mode=[1])) * Int32(2)
            low = mdO[head, d, (token_idx, batch_idx)]
            high = mdO[head + Int32(1), d, (token_idx, batch_idx)]
            fill_words[i] = _dq_pack_bf16x2(Float32(low), Float32(high))
        cute.copy(tiled_r2t, fill_values, fill_destination)
        cute.arch.fence_view_async_tmem_store()
        _tcgen05_fence_before_thread_sync()

    @cute.jit
    def _issue_dkv_sweep_hilo(
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
        high_fragment_0: cute.Tensor,
        high_fragment_1: cute.Tensor,
        low_fragment_0: cute.Tensor,
        low_fragment_1: cute.Tensor,
        first_accumulate: cutlass.Constexpr[bool],
        use_low: cutlass.Constexpr[bool],
        wait_low_on_first_chunk: cutlass.Constexpr[bool],
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
    ):
        """Consume one A sweep against BF16 high and residual-low B terms."""
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
            high_fragment = high_fragment_0 if head_base < self.H_TILE_CTA else high_fragment_1
            low_fragment = low_fragment_0 if head_base < self.H_TILE_CTA else low_fragment_1
            b_k_half = head_base % self.H_TILE_CTA // self.ROUND_K_HEADS
            accumulate = True if chunk > 0 else first_accumulate
            round_pipeline.consumer_wait(round_consumer_state)
            self._issue_dkv_pass(dkv_tiled_mma, t_dkv, round_slot_fragments[chunk], high_fragment, b_k_half, accumulate)
            if cutlass.const_expr(use_low and wait_low_on_first_chunk and chunk == 0):
                pds_pipeline.consumer_wait(pds_consumer_state)
            if cutlass.const_expr(use_low):
                self._issue_dkv_pass(dkv_tiled_mma, t_dkv, round_slot_fragments[chunk], low_fragment, b_k_half, True)
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
        return round_consumer_state

    @cute.jit
    def _issue_grads_early_dq_hilo(
        self,
        dq_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        dkv_ts_tiled_mma: cute.TiledMma,
        dot_tmem_fragment: Optional[cute.Tensor],
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
        rolled_fragments=None,
        ds2_pipeline=None,
        ds2_consumer_state=None,
        ds2_fragment_0=None,
        ds2_fragment_1=None,
        dot_tmem_fragment_1=None,
        tile_iter=None,
        tile_total=None,
    ):
        """Accumulate high/residual P products for dV and BF16 dS products for dK into FP32 TMEM.

        With ``rolled_fragments`` (stage-indexed ring, P and dS views) the ring sweeps run as loops
        (``_issue_dkv_sweep_rolled``). With ``ds2_pipeline`` (``_use_ds2`` plans) the dK products
        read the dedicated dS blocks; the P blocks (and the P staging in ``ds_image``) are released
        once the dV and dQ products are done.
        """
        if cutlass.const_expr(ds2_pipeline is not None):
            return self._issue_grads_ds2(
                dq_tiled_mma,
                dkv_tiled_mma,
                dkv_ts_tiled_mma,
                dot_tmem_fragment,
                t_dq_0,
                t_dq_1,
                t_dkv_0,
                t_dkv_1,
                dq_kd_fragment_a,
                dq_kd_fragment_b,
                dq_ds_fragment,
                round_fragment_0,
                round_fragment_1,
                round_fragment_2,
                round_fragment_3,
                p_fragment_0,
                p_fragment_1,
                ds_fragment_0,
                ds_fragment_1,
                ds2_fragment_0,
                ds2_fragment_1,
                dq_accumulate,
                relay_phase,
                relay_mbars,
                ds_local_ready_mbar,
                round_pipeline,
                round_consumer_state,
                kscore_pipeline,
                kscore_consumer_state,
                dq0_done_pipeline,
                dq0_done_state,
                dq1_done_pipeline,
                dq1_done_state,
                commit_final,
                pds_pipeline,
                pds_consumer_state,
                ds2_pipeline,
                ds2_consumer_state,
                dkv_done_pipeline,
                dkv_acquire_state,
                dkv_commit_state,
                dot_tmem_fragment_1,
                tile_iter,
                tile_total,
            )
        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        if cutlass.const_expr(rolled_fragments is not None):
            round_consumer_state = self._issue_dkv_sweep_rolled(
                dkv_tiled_mma,
                t_dkv_0,
                rolled_fragments[0],
                rolled_fragments[1],
                rolled_fragments[2],
                False,
                self.USE_P_LOW,
                True,
                pds_pipeline,
                pds_consumer_state,
                round_pipeline,
                round_consumer_state,
            )
        else:
            round_consumer_state = self._issue_dkv_sweep_hilo(
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
                ds_fragment_0,
                ds_fragment_1,
                False,
                self.USE_P_LOW,
                True,
                pds_pipeline,
                pds_consumer_state,
                round_pipeline,
                round_consumer_state,
            )
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        if cutlass.const_expr(rolled_fragments is not None):
            round_consumer_state = self._issue_dkv_sweep_rolled(
                dkv_tiled_mma,
                t_dkv_1,
                rolled_fragments[0],
                rolled_fragments[1],
                rolled_fragments[2],
                False,
                self.USE_P_LOW,
                False,
                pds_pipeline,
                pds_consumer_state,
                round_pipeline,
                round_consumer_state,
            )
        else:
            round_consumer_state = self._issue_dkv_sweep_hilo(
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
                ds_fragment_0,
                ds_fragment_1,
                False,
                self.USE_P_LOW,
                False,
                pds_pipeline,
                pds_consumer_state,
                round_pipeline,
                round_consumer_state,
            )
        pds_pipeline.consumer_release(pds_consumer_state)
        pds_consumer_state.advance()
        # Both CTAs' dS images (the dQ operand) are in place; the dS blocks follow for the dK sweeps.
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
        pds_pipeline.consumer_wait(pds_consumer_state)
        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
        if cutlass.const_expr(rolled_fragments is not None):
            round_consumer_state = self._issue_dkv_sweep_rolled(
                dkv_tiled_mma,
                t_dkv_0,
                rolled_fragments[0],
                rolled_fragments[2],
                rolled_fragments[2],
                True,
                False,
                False,
                pds_pipeline,
                pds_consumer_state,
                round_pipeline,
                round_consumer_state,
            )
        else:
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
        if cutlass.const_expr(rolled_fragments is not None):
            round_consumer_state = self._issue_dkv_sweep_rolled(
                dkv_tiled_mma,
                t_dkv_1,
                rolled_fragments[0],
                rolled_fragments[2],
                rolled_fragments[2],
                True,
                False,
                False,
                pds_pipeline,
                pds_consumer_state,
                round_pipeline,
                round_consumer_state,
            )
        else:
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
    def _issue_dv_tmem_round(
        self,
        dkv_ts_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        a_fragment: cute.Tensor,
        high_fragment_0: cute.Tensor,
        high_fragment_1: cute.Tensor,
        low_fragment_0: cute.Tensor,
        low_fragment_1: cute.Tensor,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        wait_low: cutlass.Constexpr[bool],
    ):
        """One D-round of dV from its TMEM dO^T operand: the high chunks, then the residual-low chunks.

        ``wait_low`` waits for the P_lo blocks before the low terms (the first D-round of a tile).
        """
        chunk_k_blocks = self.ROUND_K_HEADS // 16
        for term in cutlass.range_constexpr(2):
            if cutlass.const_expr(term == 1 and wait_low):
                pds_pipeline.consumer_wait(pds_consumer_state)
            for chunk in cutlass.range_constexpr(self.ROUND_STAGES):
                head_base = chunk * self.ROUND_K_HEADS
                b_fragment = high_fragment_0 if head_base < self.H_TILE_CTA else high_fragment_1
                if cutlass.const_expr(term == 1):
                    b_fragment = low_fragment_0 if head_base < self.H_TILE_CTA else low_fragment_1
                b_k_block_offset = head_base % self.H_TILE_CTA // self.ROUND_K_HEADS * chunk_k_blocks
                mma = dkv_ts_tiled_mma.with_()
                mma.set(tcgen05.Field.ACCUMULATE, term > 0 or chunk > 0)
                for k_block in cutlass.range_constexpr(chunk_k_blocks):
                    cute.gemm(
                        mma, t_dkv, a_fragment[None, None, chunk * chunk_k_blocks + k_block], b_fragment[None, None, b_k_block_offset + k_block, 0], t_dkv
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_grads_ds2(
        self,
        dq_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        dkv_ts_tiled_mma: cute.TiledMma,
        dot_tmem_fragment_0: cute.Tensor,
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
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        ds2_fragment_0: cute.Tensor,
        ds2_fragment_1: cute.Tensor,
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
        ds2_pipeline,
        ds2_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dkv_commit_state: pipeline.PipelineState,
        dot_tmem_fragment_1=None,
        tile_iter=None,
        tile_total=None,
    ):
        """``_issue_grads_early_dq_hilo`` for ``_use_ds2`` plans: both dO^T operands in TMEM, dK from the dS blocks.

        Such plans hold the D-round-0 and D-round-1 dO^T operands in TMEM (``DV0_TMEM_A``,
        ``DV1_TMEM_A``) and one dKV accumulator for both D-rounds, and alternate the D-round order per
        tile (``ROUND_REUSE``): tile ``tile_iter`` sweeps D-round ``tile_iter % 2`` first. The tile's
        first Q^T group is the one its predecessor swept last and is still in the ring (awaited only by
        tile 0, released here); its second group is awaited and kept for the next tile (released only by
        the last tile).

        The first D-round (dV from TMEM, dQ, dK from the Q^T ring) is committed to the reducers; the
        second then reacquires the same accumulator columns once they were loaded, adds its dV from TMEM
        (the P phase is released after it) and its dK from the ring. Each accumulator keeps one fixed
        FP32 order (dV high chunks, dV low chunks, dK chunks). The dS blocks (``ds2``) are released by
        the caller after the dK products.
        """
        assert self.DV1_TMEM_A and self.ROUND_REUSE, "dedicated-dS plans hold both dO^T operands in TMEM and alternate the D-rounds"
        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        if tile_iter % Int32(2) == Int32(0):
            self._issue_dv_tmem_round(
                dkv_ts_tiled_mma, t_dkv_0, dot_tmem_fragment_0, p_fragment_0, p_fragment_1, ds_fragment_0, ds_fragment_1, pds_pipeline, pds_consumer_state, True
            )
        else:
            self._issue_dv_tmem_round(
                dkv_ts_tiled_mma, t_dkv_0, dot_tmem_fragment_1, p_fragment_0, p_fragment_1, ds_fragment_0, ds_fragment_1, pds_pipeline, pds_consumer_state, True
            )
        # Both CTAs' dS images (the dQ operand) are in place.
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
        ds2_pipeline.consumer_wait(ds2_consumer_state)
        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
        # First Q^T group: generation block ``tile_iter``.
        self._issue_dkv_sweep_reuse(
            dkv_tiled_mma,
            t_dkv_0,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            ds2_fragment_0,
            ds2_fragment_1,
            round_pipeline,
            Int32(self.ROUND_STAGES) * tile_iter,
            tile_iter == Int32(0),
            True,
        )
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        if tile_iter % Int32(2) == Int32(0):
            self._issue_dv_tmem_round(
                dkv_ts_tiled_mma,
                t_dkv_1,
                dot_tmem_fragment_1,
                p_fragment_0,
                p_fragment_1,
                ds_fragment_0,
                ds_fragment_1,
                pds_pipeline,
                pds_consumer_state,
                False,
            )
        else:
            self._issue_dv_tmem_round(
                dkv_ts_tiled_mma,
                t_dkv_1,
                dot_tmem_fragment_0,
                p_fragment_0,
                p_fragment_1,
                ds_fragment_0,
                ds_fragment_1,
                pds_pipeline,
                pds_consumer_state,
                False,
            )
        pds_pipeline.consumer_release(pds_consumer_state)
        pds_consumer_state.advance()
        # Second Q^T group: generation block ``tile_iter + 1``, kept in the ring for the next tile.
        self._issue_dkv_sweep_reuse(
            dkv_tiled_mma,
            t_dkv_1,
            round_fragment_0,
            round_fragment_1,
            round_fragment_2,
            round_fragment_3,
            ds2_fragment_0,
            ds2_fragment_1,
            round_pipeline,
            Int32(self.ROUND_STAGES) * (tile_iter + Int32(1)),
            True,
            tile_iter == tile_total - Int32(1),
        )
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        return (round_consumer_state, kscore_consumer_state, dkv_acquire_state, dkv_commit_state, pds_consumer_state, ds2_consumer_state)

    @cute.jit
    def _drain_dkv_deterministic(
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
        split_wg_shard: cutlass.Boolean,
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):
        """Drain dKV with two disjoint warp-group planes when capacity permits."""
        seqlen_kv = cute.size(mdKV_acc, mode=[1])
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        pitch = cute.round_up(seqlen_kv, 8)
        wg_offset = cutlass.Int64(wg_idx * Int32(split_wg_shard)) * cutlass.Int64(pitch) * cutlass.Int64(self.head_dim)
        mdKV_acc = cute.make_tensor(mdKV_acc.iterator + wg_offset, mdKV_acc.layout)
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
        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
        kv_index = Int32(-1)
        owner_lane = Int32(0)
        # Cluster ranks own disjoint D128 column slices. Within each slice,
        # duplicate-index writers take turns by warp group and then lane group.
        # Separate warp-group planes need only the lane-group rendezvous.
        for owner_wg in cutlass.range(2):
            if (split_wg_shard and owner_wg == Int32(0)) or (split_wg_shard == cutlass.Boolean(False) and wg_idx == owner_wg):
                for owner_lane in cutlass.range(4):
                    if dp_idx % Int32(4) == owner_lane:
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
                                ordered_dkv_quad(target_frg_0, rdkv_frg_0)
                    cute.arch.sync_warp()
            if split_wg_shard == cutlass.Boolean(False) and owner_wg == Int32(0):
                self.dsink_reducer_barrier.arrive_and_wait()
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            done_pipeline.consumer_release(release_state)
        release_state.advance()
        kv_index = Int32(-1)
        owner_lane = Int32(0)
        for owner_wg in cutlass.range(2):
            if (split_wg_shard and owner_wg == Int32(0)) or (split_wg_shard == cutlass.Boolean(False) and wg_idx == owner_wg):
                for owner_lane in cutlass.range(4):
                    if dp_idx % Int32(4) == owner_lane:
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
                                ordered_dkv_quad(target_frg_1, rdkv_frg_1)
                    cute.arch.sync_warp()
            if split_wg_shard == cutlass.Boolean(False) and owner_wg == Int32(0):
                self.dsink_reducer_barrier.arrive_and_wait()
        return (wait_state, release_state)

    @cute.kernel
    def fold_groups_convert_dkv_finalize_dSink_deterministic(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        active_groups: Int32,
        sum_odo: cute.Tensor,
        d_sink: cute.Tensor,
        problem_shape,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Finish the group fold, emit BF16, and publish the stream-ordered dSink fold."""
        row, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if row < seqlen:
            pitch = cute.round_up(seqlen, 8)
            source_base = tidx * Int32(16)
            final_values = cute.make_rmem_tensor((16,), Float32)
            for value_idx in cutlass.range_constexpr(16):
                final_values[value_idx] = Float32(0.0)
            for group in cutlass.range(active_groups):
                acc_row = mdKV_acc[None, row + group * Int32(self.dkv_fold_group_size) * pitch, (0, 0)]
                source = acc_row.iterator + source_base
                loaded = (*_load_f32x4(source), *_load_f32x4(source + Int32(4)), *_load_f32x4(source + Int32(8)), *_load_f32x4(source + Int32(12)))
                for value_idx in cutlass.range_constexpr(16):
                    final_values[value_idx] = final_values[value_idx] + loaded[value_idx]
            out_row = mdKV[None, row, (0, 0)]
            chunk = source_base // Int32(32) * Int32(32)
            a0 = source_base % Int32(32) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = _dq_pack_bf16x2(final_values[b], final_values[b + 4])
                packed23 = _dq_pack_bf16x2(final_values[b + 8], final_values[b + 12])
                _store_bf16x4_ordinary(out_row.iterator + output_base, packed01, packed23)
        if row < min(seqlen, Int32(4)):
            head = tidx + Int32(32) * row
            while head < problem_shape[3][0]:
                self._finalize_dSink_head_deterministic(sum_odo, d_sink, problem_shape, head, chunk_size)
                head += Int32(32) * seqlen

    @cute.kernel
    def fold_convert_dkv_deterministic(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen: Int32, active_shards: Int32):
        """Fold the fixed shard tree and directly emit the established BF16 layout."""
        seq_block, _, _ = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_id = seq_block * Int32(4) + tidy
        assert self.head_dim == 512
        assert self.dkv_fold_group_size == 8
        if seq_id < seqlen:
            pitch = cute.round_up(seqlen, 8)
            active_groups = cute.ceil_div(active_shards, self.dkv_fold_group_size)
            source_base = tidx * Int32(16)
            final_values = cute.make_rmem_tensor((16,), Float32)
            partial_values = cute.make_rmem_tensor((16,), Float32)
            for value_idx in cutlass.range_constexpr(16):
                final_values[value_idx] = Float32(0.0)
            for group in cutlass.range(active_groups):
                for value_idx in cutlass.range_constexpr(16):
                    partial_values[value_idx] = Float32(0.0)
                group_begin = group * Int32(self.dkv_fold_group_size)
                for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                    shard = group_begin + Int32(local_shard)
                    if shard < active_shards:
                        acc_row = mdKV_acc[None, seq_id + shard * pitch, (0, 0)]
                        source = acc_row.iterator + source_base
                        for half in cutlass.range_constexpr(2):
                            loaded = _load_f32x8_cg(source + Int32(half * 8))
                            for lane in cutlass.range_constexpr(8):
                                value_idx = half * 8 + lane
                                partial_values[value_idx] = partial_values[value_idx] + loaded[lane]
                for value_idx in cutlass.range_constexpr(16):
                    final_values[value_idx] = final_values[value_idx] + partial_values[value_idx]
            out_row = mdKV[None, seq_id, (0, 0)]
            chunk = source_base // Int32(32) * Int32(32)
            a0 = source_base % Int32(32) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = _dq_pack_bf16x2(final_values[b], final_values[b + 4])
                packed23 = _dq_pack_bf16x2(final_values[b + 8], final_values[b + 12])
                _store_bf16x4_ordinary(out_row.iterator + output_base, packed01, packed23)

    @cute.kernel
    def fold_convert_dkv_finalize_dSink_deterministic(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        active_shards: Int32,
        sum_odo: cute.Tensor,
        d_sink: cute.Tensor,
        problem_shape,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Fold/convert large extents and hide the one-writer dSink final fold."""
        seq_block, _, _ = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_id = seq_block * Int32(4) + tidy
        assert self.head_dim == 512
        assert self.dkv_fold_group_size == 8
        if seq_id < seqlen:
            pitch = cute.round_up(seqlen, 8)
            active_groups = cute.ceil_div(active_shards, self.dkv_fold_group_size)
            source_base = tidx * Int32(16)
            final_values = cute.make_rmem_tensor((16,), Float32)
            partial_values = cute.make_rmem_tensor((16,), Float32)
            for value_idx in cutlass.range_constexpr(16):
                final_values[value_idx] = Float32(0.0)
            for group in cutlass.range(active_groups):
                for value_idx in cutlass.range_constexpr(16):
                    partial_values[value_idx] = Float32(0.0)
                group_begin = group * Int32(self.dkv_fold_group_size)
                for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                    shard = group_begin + Int32(local_shard)
                    if shard < active_shards:
                        acc_row = mdKV_acc[None, seq_id + shard * pitch, (0, 0)]
                        source = acc_row.iterator + source_base
                        for half in cutlass.range_constexpr(2):
                            loaded = _load_f32x8_cg(source + Int32(half * 8))
                            for lane in cutlass.range_constexpr(8):
                                value_idx = half * 8 + lane
                                partial_values[value_idx] = partial_values[value_idx] + loaded[lane]
                for value_idx in cutlass.range_constexpr(16):
                    final_values[value_idx] = final_values[value_idx] + partial_values[value_idx]
            out_row = mdKV[None, seq_id, (0, 0)]
            chunk = source_base // Int32(32) * Int32(32)
            a0 = source_base % Int32(32) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = _dq_pack_bf16x2(final_values[b], final_values[b + 4])
                packed23 = _dq_pack_bf16x2(final_values[b + 8], final_values[b + 12])
                _store_bf16x4_ordinary(out_row.iterator + output_base, packed01, packed23)
        if seq_block == Int32(0):
            head = tidx + Int32(32) * tidy
            self._finalize_dSink_head_deterministic(sum_odo, d_sink, problem_shape, head, chunk_size)


class FlashAttentionDSABackwardSm100H128TwoCTADeterministicRankShard(RankShardReduction, FlashAttentionDSABackwardSm100H128TwoCTADeterministic):
    """The deterministic H128 D512 two-CTA kernel with conflict-rank dKV storage (``RankShardReduction``).

    This class runs one setup launch before each main kernel. ``...RankShardFewPlanes`` is the same
    plan with at most ``RANK_FEW_PLANES`` dense planes; ``...RankShardFused`` runs each wave's tail in
    the next main kernel's reducer warps and the slot phases in its setup warp.

    TMEM: S(t + 1) and dP(t + 1) issue only after the dS of tile t was read, so one score and one
    dP buffer suffice (``SCORE_DONE_STAGES = 1``). Top-k 512 and wider hold the D-round-0 dO^T
    operand of dV in the freed columns (``DV0_TMEM_A``) and read dK's dS from dedicated blocks
    (``_use_ds2``); the dKV accumulator then keeps one D-round at a time (``MMA_DONE_STAGES = 1``),
    the freed D-round-1 accumulator columns hold the D-round-1 dO^T (``DV1_TMEM_A``), the dK/dV
    ring streams only Q^T, and tiles alternate their D-round order (``ROUND_REUSE``).
    """

    rank_record = 512
    # The compiled workspace view is one record-pitched plane of dynamic extent.
    num_dkv_shards = 1
    # S(t + 1) and dP(t + 1) issue only after the dS of tile t was read, so one TMEM buffer each suffices; the
    # freed columns hold the D-round-0 dO^T operand of dV, which then leaves the dK/dV ring stream.
    SCORE_DONE_STAGES = 1
    TMEM_S1_OFFSET = 0
    TMEM_DP_OFFSET = 32
    TMEM_DP1_OFFSET = 32
    TMEM_A_OFFSET = 64
    DV0_TMEM_A = True
    PLO_STAGING_MAX_TOPK = 128

    def __init__(self, element_dtype: Type[cutlass.Numeric], head_dim: int, head_dim_v: int, block_tile: int, max_topk: int = 0):
        super().__init__(element_dtype, head_dim, head_dim_v, block_tile, max_topk)
        self.num_dkv_shards = 1
        self.split_wg_queries_per_wave = 0
        if max_topk <= 128:
            # Two-tile queries stream every round stage: staging dO^T in TMEM costs more per query than the
            # four ring stages per tile it saves (the single-buffered score/dP accumulators stay).
            self.DV0_TMEM_A = False
            self.DS_RELAY_ON_RELEASE = True
        if self._use_ds2():
            assert self.ROUND_STAGES == 4, "dedicated-dS plans stream four ring stages per D-round group"
            # Dedicated-dS plans hold both D-rounds' dO^T operands in TMEM: the dKV accumulator keeps one
            # D-round at a time (D-round 1 reuses the round-0 columns once the reducers loaded them), so the
            # D-round-1 dO^T takes the freed columns and the dK/dV ring streams only Q^T (8 stages a tile).
            self.DV1_TMEM_A = True
            self.MMA_DONE_STAGES = 1
            self.TMEM_DKV1_OFFSET = self.TMEM_DKV0_OFFSET
            # Alternate the D-round order per tile and keep the shared Q^T group in the ring.
            self.ROUND_REUSE = max_topk >= self.ROUND_REUSE_MIN_TOPK


class FlashAttentionDSABackwardSm100H128TwoCTADeterministicRankShardFewPlanes(FlashAttentionDSABackwardSm100H128TwoCTADeterministicRankShard):
    """Per-wave-setup conflict-rank plan with at most ``RANK_FEW_PLANES`` (three) dense planes.

    Most rows that several queries of a wave share overflow there, so most drained slots carry
    indirect codes: the drain overlaps their overflow-offset lookups (``RANK_DRAIN_OVERLAP_LOOKUPS``).
    """

    rank_few_planes = True
    RANK_DRAIN_OVERLAP_LOOKUPS = True
    RANK_RECORDS_L2_DEFAULT = True


class FlashAttentionDSABackwardSm100H128TwoCTADeterministicRankShardFused(FlashAttentionDSABackwardSm100H128TwoCTADeterministicRankShard):
    """Conflict-rank storage whose wave tails run inside the next main kernel (no per-wave setup launch)."""

    rank_fused = True
