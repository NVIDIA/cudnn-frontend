# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the conflict-rank dKV storages of the deterministic SM100 M64 DSA backward kernel.

The records storage (``dsa_bwd_sm100_rank_shard_m64``), the chain relay (``dsa_bwd_sm100_rank_chain_m64``)
and the main kernel's conflict-rank reducers use these PTX wrappers (vector loads and stores, L2-hinted
reductions, GPU-scope release/acquire flags, L2 prefetch), the clamped top-k extent, the plane geometry and
the constants of the records plane rule, which both storages size against. The H128 two-CTA kernels use
its subset barrier (``_partial_cta_sync``) from their warp-role branches.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op

# Query CTAs per records wave: one per SM of B200, a plan constant that does not follow the device's SM count;
# correctness needs nothing but the two CTAs of a cluster to be co-resident, on any SM count.
RANK_M64_WAVE = 148
# Plane policy of the records rule (``dsa_bwd_sm100_rank_shard_m64.rank_m64_planes``): at most about 1 GiB of
# dense planes, a second plane within 1.5 GiB, and the margin dropped above ``RANK_M64_COMPACT_BYTES``.
RANK_M64_DENSE_BUDGET_BYTES = 1 << 30
RANK_M64_PAIR_BUDGET_BYTES = 3 << 29
RANK_M64_COMPACT_BYTES = (1 << 30) - (64 << 20)
# Row-state words per KV row and wave slot of the records rule: five ordinal words, a duplicate flag, padding.
RANK_M64_STATE_WORDS = 8
# Grid and block of the small setup kernels.
RANK_M64_SETUP_CTAS = 296
RANK_M64_THREADS = 256
# Destination-code flag: the slot stores instead of adding (first writer of its plane row or record).
RANK_M64_FRESH_FLAG = 1 << 30


def rank_m64_pitch(total_s_kv: int) -> int:
    """Return the padded KV row count of one plane."""
    return (int(total_s_kv) + 7) // 8 * 8


def rank_m64_record(head_dim: int) -> int:
    """Return the FP32 width of one dKV record."""
    return (int(head_dim) + 7) // 8 * 8


@dsl_user_op
def _partial_cta_sync(barrier_id: int, thread_count: int, *, loc=None, ip=None):
    """Synchronize a named-barrier subset without an aligned-CTA promise."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(thread_count).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rank_load_u32x4(source: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned 32-bit words with one 128-bit load."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _rank_store_zero_x4(destination: cute.Pointer, *, loc=None, ip=None) -> None:
    """Clear four aligned 32-bit words with one 128-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None, [destination_i64], "st.global.v4.b32 [$0], {0, 0, 0, 0};", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


@dsl_user_op
def _rank_load_f32x4(source: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned FP32 values with one 128-bit load."""
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
def _rank_store_f32x4(destination: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Store four aligned FP32 values with one 128-bit store."""
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
def _rank_store_f32x2(destination: cute.Pointer, value_0: Float32, value_1: Float32, *, loc=None, ip=None) -> None:
    """Store two aligned FP32 values with one 64-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Float32(value_0).ir_value(loc=loc, ip=ip), Float32(value_1).ir_value(loc=loc, ip=ip)],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_red_f32x4(destination: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Add four FP32 values to an accumulating row with an L2 evict_first policy (the row is not revisited this wave)."""
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
        "{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 policy, 1.0; red.global.add.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, policy; }",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_red_f32x2(destination: cute.Pointer, value_0: Float32, value_1: Float32, *, loc=None, ip=None) -> None:
    """Add two FP32 values to an accumulating row with an L2 evict_first policy."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Float32(value_0).ir_value(loc=loc, ip=ip), Float32(value_1).ir_value(loc=loc, ip=ip)],
        "{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 policy, 1.0; red.global.add.L2::cache_hint.v2.f32 [$0], {$1, $2}, policy; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_load_once_f32x4(source: cute.Pointer, *, loc=None, ip=None):
    """Load four FP32 values that are read once (records): evict them from L2 first."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "{ .reg .b64 policy; createpolicy.fractional.L2::evict_first.b64 policy, 1.0; ld.global.cg.L2::cache_hint.v4.f32 {$0, $1, $2, $3}, [$4], policy; }",
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
def _rank_prefetch_l2(source: cute.Pointer, num_bytes: int, *, loc=None, ip=None) -> None:
    """Prefetch ``num_bytes`` (a 16-byte multiple at a 16-byte aligned address) of global memory into L2."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [source_i64],
        f"cp.async.bulk.prefetch.L2.global [$0], {num_bytes};",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_flag_release(flag: cute.Pointer, value: Int32, *, loc=None, ip=None) -> None:
    """Publish ``value`` after every earlier memory operation of this thread (gpu-scope release)."""
    flag_i64 = flag.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [flag_i64, Int32(value).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_flag_acquire(flag: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Read a published value; later memory operations of this thread are ordered after it (gpu-scope acquire)."""
    flag_i64 = flag.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(
        T.i32(), [flag_i64], "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )
    return Int32(value)


@cute.jit
def _rank_topk(mTopkIdxs: cute.Tensor, mTopkLength: Optional[cute.Tensor], query: Int32) -> Int32:
    """Return the clamped active top-k extent of one query."""
    topk = Int32(mTopkIdxs.shape[0])
    if cutlass.const_expr(mTopkLength is not None):
        raw = Int32(mTopkLength[query])
        if raw < topk:
            topk = raw
        if topk < Int32(0):
            topk = Int32(0)
    return topk


class ConflictRankReducerMixin:
    """Reducer helpers shared by the records and chain kernel classes (``RankShardM64Reduction``, ``RankChainM64Reduction``)."""

    @cute.jit
    def _prefetch_dKV_destinations(
        self,
        mTopkIdxs: cute.Tensor,
        query_codes: cute.Pointer,
        tTR_cdKV: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        full_tiles,
        token_idx: Int32,
        batch_idx: Int32,
        rNextIdx: cute.Tensor,
        rNextCode: cute.Tensor,
    ):
        """Issue the index and destination-code loads of one tile's eight reducer rows (conflict-rank plans)."""
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            local_row_idx = cute.get(tTR_cdKV[coord_base], mode=[1])
            global_row_idx = tile_index * self.block_tile + local_row_idx
            if full_tiles:
                rNextIdx[i] = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                rNextCode[i] = cute.arch.load(query_codes + Int64(global_row_idx), Int32)
            else:
                rNextIdx[i] = Int32(-1)
                rNextCode[i] = Int32(0)
                if global_row_idx < topk:
                    rNextIdx[i] = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                    rNextCode[i] = cute.arch.load(query_codes + Int64(global_row_idx), Int32)
