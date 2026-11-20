# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Base of the MXFP8 backward megakernel: the task kinds and warp roles, the BF16 GEMM task of the
shared expert (geometry, operand waits, the mainloop on a six-stage A/B ring with one accumulator
stage, the epilogue), the CLC drain of the grid's capacity tail, and the kernel zeroing the weight
gradients of experts without rows.

A GEMM task waits on the readiness counters for its operands -- a dgrad before its first K tile, a
Wgrad as a K tile first touches a 256-row block or a minibatch -- and arrives on them after its
stores complete.  All sizes are runtime values; ``has_tail`` and the clamp's presence are compile
time.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float32, Int32, Int64
from . import _mxfp8_gemm_base as gemm
from ._bwd_device_helpers import (
    _mbar_spin_wait,
    _counter_wait_gpu,
    _counter_arrive_gpu,
    _ceil_div_i32,
    _swizzled_tile_i32,
)

THREADS_PER_CTA = 256
EPILOGUE_WARP_LAST = 3
UMMA_PRODUCER_WARP = 4
CLC_PRODUCER_WARP = 5
BF16_IDLE_WARP = 6
AB_TMA_PRODUCER_WARP = 7
MLP_TILE_M = 256
MLP_TILE_K = 64
SWIGLU_TILE_M = 128
SWIGLU_TILE_N = 128
SWIGLU_TMA_TRANSACTION_BYTES = 98304
CLC_RESPONSE_BYTES = 16

SHARED_DOWN_DGRAD = 0
SHARED_SWIGLU_BWD = 1
SHARED_GATE_UP_DGRAD = 2
SHARED_WGRAD_DOWN = 3
SHARED_WGRAD_GATE = 4
SHARED_WGRAD_UP = 5
REPLAY_GATE = 6
REPLAY_UP = 7
# 8 is REPLAY_SWIGLU (the MXFP8 backward defines it).
ROUTED_DOWN_DGRAD = 9
ROUTED_SWIGLU_BWD = 10
ROUTED_GATE_UP_DGRAD = 11
ROUTED_WGRAD_DOWN = 12
ROUTED_WGRAD_GATE = 13
ROUTED_WGRAD_UP = 14


@cute.jit
def _expert_begin(counts: cute.Tensor, expert: Int32) -> Int32:
    begin = Int32(0)
    for index in cutlass.range(expert):
        begin += Int32(counts[index])
    return begin


@cute.jit
def _expert_for_token_prefix(prefix: cute.Tensor, token: Int32, experts: Int32):
    """upper_bound over the token prefix: expert with prefix[e] <= token < prefix[e+1]."""
    lo = Int32(0)
    hi = experts
    while hi - lo > Int32(1):
        mid = (lo + hi) // Int32(2)
        if prefix[mid] <= token:
            lo = mid
        else:
            hi = mid
    return lo, prefix[lo], prefix[lo + Int32(1)]


@cute.jit
def _expert_for_token(counts: cute.Tensor, token: Int32, experts: Int32):
    expert = Int32(0)
    begin = Int32(0)
    end = Int32(counts[0])
    while expert + Int32(1) < experts and end <= token:
        expert += Int32(1)
        begin = end
        end += Int32(counts[expert])
    return expert, begin, end


@cute.jit
def _is_wgrad_kind(kind: Int32) -> Int32:
    result = Int32(0)
    if kind >= Int32(SHARED_WGRAD_DOWN) and kind <= Int32(SHARED_WGRAD_UP):
        result = Int32(1)
    elif kind >= Int32(ROUTED_WGRAD_DOWN):
        result = Int32(1)
    return result


class _Mxfp8BwdBase:
    # Compile time: H or I may end in a 128-wide tail tile; the MXFP8 backward sets it per variant.
    has_tail = True

    def __init__(self, *, is_clamped: bool) -> None:
        # Only the presence of a clamp is compile time, never its value.
        self.is_clamped = is_clamped
        self.abt = gemm._Bf16Gemm()
        self.atb = gemm._Bf16Gemm()

    @cute.jit
    def _gemm_geometry(
        self,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        local: Int32,
        num_tokens: Int32,
        tokens_per_expert: cute.Tensor,
        expert_prefix: cute.Tensor,
        expert_valid_end: cute.Tensor,
        use_prefix: Int32,
        *,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ):
        """Return valid, expert, macro-local M/N tile, global K range, first."""
        if cutlass.const_expr(self.has_tail):
            intermediate_blocks = _ceil_div_i32(intermediate_size, Int32(256))
            hidden_blocks = _ceil_div_i32(hidden_size, Int32(256))
        else:
            intermediate_blocks = intermediate_size // Int32(256)
            hidden_blocks = hidden_size // Int32(256)
        weight_tiles = intermediate_blocks * hidden_blocks
        valid = Int32(0)
        expert = Int32(0)
        row = Int32(0)
        column = Int32(0)
        k_start = Int32(0)
        k_end = Int32(0)
        first = Int32(1)
        columns = intermediate_blocks
        if kind == Int32(SHARED_GATE_UP_DGRAD) or kind == Int32(ROUTED_GATE_UP_DGRAD):
            columns = hidden_blocks
        if _is_wgrad_kind(kind) != Int32(0):
            rows = intermediate_blocks
            columns = hidden_blocks
            if kind == Int32(SHARED_WGRAD_DOWN) or kind == Int32(ROUTED_WGRAD_DOWN):
                rows = hidden_blocks
                columns = intermediate_blocks
            if kind <= Int32(SHARED_WGRAD_UP):
                k_end = Int32(num_local_tokens)
            else:
                expert = local // weight_tiles
                local = local % weight_tiles
                expert_begin = Int32(0)
                expert_end_tokens = Int32(0)
                if use_prefix != Int32(0):
                    expert_begin = expert_prefix[expert]
                    # Skip the whole-64-row padding blocks at the end of the expert segment
                    # (padding rows are zero; the dW contribution is exact). A non-empty expert keeps
                    # >= 1 K block so it never loses its dW store even if the scheduler invariant
                    # (padding < 256 rows) were violated; an empty expert stays empty.
                    padded_end_tokens = expert_prefix[expert + Int32(1)]
                    expert_end_tokens = cutlass.max(Int32(expert_valid_end[expert]), cutlass.min(expert_begin + Int32(MLP_TILE_K), padded_end_tokens))
                else:
                    expert_begin = _expert_begin(tokens_per_expert, expert)
                    expert_end_tokens = expert_begin + Int32(tokens_per_expert[expert])
                k_start = cutlass.max(expert_begin, macro * Int32(macrobatch_size))
                macro_begin = macro * macrobatch_size
                macro_end = macro_begin + cutlass.min(macrobatch_size, num_tokens - macro_begin)
                k_end = cutlass.min(expert_end_tokens, macro_end)
                first = Int32(k_start == expert_begin)
            if k_start < k_end:
                valid = Int32(1)
                row, column = _swizzled_tile_i32(rows, columns, local)
        elif kind <= Int32(SHARED_GATE_UP_DGRAD):
            rows = Int32(num_local_tokens // MLP_TILE_M)
            if local < rows * columns:
                valid = Int32(1)
                row, column = _swizzled_tile_i32(rows, columns, local)
        else:
            mini_begin = generation * Int32(minibatch_size // MLP_TILE_M)
            mini_end = cutlass.min(mini_begin + Int32(minibatch_size // MLP_TILE_M), num_tokens // Int32(MLP_TILE_M))
            candidate_row = mini_begin + local // columns
            if candidate_row < mini_end:
                target_token = candidate_row * Int32(MLP_TILE_M)
                expert = Int32(0)
                expert_begin = Int32(0)
                expert_end = Int32(0)
                if use_prefix != Int32(0):
                    expert, expert_begin, expert_end = _expert_for_token_prefix(expert_prefix, target_token, num_local_experts)
                else:
                    expert, expert_begin, expert_end = _expert_for_token(tokens_per_expert, target_token, num_local_experts)
                begin = cutlass.max(mini_begin, expert_begin // Int32(MLP_TILE_M))
                end = cutlass.min(mini_end, expert_end // Int32(MLP_TILE_M))
                rows = end - begin
                remaining = local - (begin - mini_begin) * columns
                row, column = _swizzled_tile_i32(rows, columns, remaining)
                row = row + begin - macro * Int32(macrobatch_size // MLP_TILE_M)
                valid = Int32(1)
        return valid, expert, row, column, k_start, k_end, first

    @cute.jit
    def _drain_clc_capacity_tail(
        self,
        needs_drain: Int32,
        clc_drain_pipeline,
        drain_consumer_state,
        drain_producer_state,
        drain_scheduler,
        cta_rank: Int32,
    ) -> None:
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if needs_drain != Int32(0):
            if cute.arch.lane_idx() == Int32(0):
                # Per-warp drain stage (index fixed for the whole loop; the phase
                # flips every iteration because the state advances a full depth).
                full_bar = clc_drain_pipeline.sync_object_full.get_barrier(drain_consumer_state.index)
                empty_bar = clc_drain_pipeline.sync_object_empty.get_barrier(drain_producer_state.index)
                response = drain_scheduler._clc_response_ptr
                full_phase = drain_consumer_state.phase
                empty_phase = Int32(0)
                cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(CLC_RESPONSE_BYTES))
                if cta_rank == Int32(0):
                    cute.arch.issue_clc_query(full_bar, response)
                active = Int32(1)
                while active != Int32(0):
                    _mbar_spin_wait(full_bar, full_phase)
                    full_phase = full_phase ^ Int32(1)
                    _, _, _, drained_valid = cute.arch.clc_response(response)
                    # The generic-proxy read of the response must be ordered before the next
                    # async-proxy CLC write into the same shared memory.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    # Both CTAs release rank 0's empty barrier (cluster arrive).
                    clc_drain_pipeline.sync_object_empty.arrive(drain_consumer_state.index, clc_drain_pipeline.consumer_mask)
                    if cta_rank == Int32(0):
                        _mbar_spin_wait(empty_bar, empty_phase)
                        empty_phase = empty_phase ^ Int32(1)
                    active = Int32(drained_valid)
                    if active != Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(CLC_RESPONSE_BYTES))
                        if cta_rank == Int32(0):
                            cute.arch.issue_clc_query(full_bar, response)
            cute.arch.sync_warp()

    @cute.jit
    def _wait_task_operands(
        self,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        row: Int32,
        num_tokens: Int32,
        d_y_ready,
        d_gate_up_ready,
        replayed_x_ready,
        *,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ):
        swiglu_columns = intermediate_size // Int32(128)
        dispatch_columns = _ceil_div_i32(hidden_size, Int32(512))
        if kind == Int32(REPLAY_GATE) or kind == Int32(REPLAY_UP) or kind == Int32(ROUTED_DOWN_DGRAD):
            rows = cutlass.min(Int32(minibatch_size), num_tokens - generation * Int32(minibatch_size))
            required = _ceil_div_i32(rows, Int32(128)) * dispatch_columns
            if kind == Int32(ROUTED_DOWN_DGRAD):
                _counter_wait_gpu(d_y_ready, generation, required)
            else:
                _counter_wait_gpu(replayed_x_ready, generation, required)
        elif kind == Int32(SHARED_GATE_UP_DGRAD):
            _counter_wait_gpu(d_gate_up_ready, row, swiglu_columns * Int32(2))
        elif kind == Int32(ROUTED_GATE_UP_DGRAD):
            _counter_wait_gpu(d_gate_up_ready, Int32(num_local_tokens // 256) + macro * Int32(macrobatch_size // 256) + row, swiglu_columns * Int32(2))

    @cute.jit
    def _wait_wgrad_operands(
        self,
        kind: Int32,
        macro: Int32,
        wait_row_block: Int32,
        wait_minibatch: Int32,
        row: Int32,
        num_tokens: Int32,
        d_y_ready,
        d_gate_up_ready,
        replayed_x_ready,
        *,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        minibatch_size: Int32,
    ):
        swiglu_columns = intermediate_size // Int32(128)
        dispatch_columns = _ceil_div_i32(hidden_size, Int32(512))
        if wait_row_block != Int32(0):
            if kind == Int32(SHARED_WGRAD_GATE) or kind == Int32(SHARED_WGRAD_UP):
                _counter_wait_gpu(d_gate_up_ready, row // Int32(256), swiglu_columns * Int32(2))
            elif kind >= Int32(ROUTED_WGRAD_DOWN):
                # Routed Wgrad-down/gate/up all wait for the row block's SwiGLU-backward.
                _counter_wait_gpu(d_gate_up_ready, Int32(num_local_tokens // 256) + row // Int32(256), swiglu_columns * Int32(2))
        if wait_minibatch != Int32(0):
            mini = row // Int32(minibatch_size)
            rows = cutlass.min(Int32(minibatch_size), num_tokens - mini * Int32(minibatch_size))
            required = _ceil_div_i32(rows, Int32(128)) * dispatch_columns
            if kind == Int32(ROUTED_WGRAD_DOWN):
                _counter_wait_gpu(d_y_ready, mini, required)
            elif (kind == Int32(ROUTED_WGRAD_GATE) or kind == Int32(ROUTED_WGRAD_UP)) and macro > Int32(0):
                _counter_wait_gpu(replayed_x_ready, mini, required)

    @cute.jit
    def _run_one_gemm(
        self,
        collective,
        tiled_mma,
        a_atom,
        a_tensor,
        b_atom,
        b_tensor,
        a2_atom,
        a2_tensor,
        b2_atom,
        b2_tensor,
        d_atom,
        d_tensor,
        sA,
        sB,
        sD,
        epi_tile,
        acc_ptr,
        ab_pipeline,
        acc_pipeline,
        ab_producer_state,
        ab_consumer_state,
        acc_producer_state,
        acc_consumer_state,
        thread: Int32,
        warp: Int32,
        cta_rank: Int32,
        kind: Int32,
        macro: Int32,
        generation: Int32,
        row: Int32,
        column: Int32,
        k_start: Int32,
        k_end: Int32,
        num_tokens: Int32,
        d_y_ready,
        d_gate_up_ready,
        replayed_x_ready,
        d_hidden_ready,
        d_x_ready,
        replayed_gate_up_ready,
        routed_buffers_done,
        is_wgrad: cutlass.Constexpr,
        second: cutlass.Constexpr,
        d_add_atom=None,
        d_add_tensor=None,
        first_contribution: Int32 = 1,
        *,
        num_local_tokens: Int32,
        hidden_size: Int32,
        intermediate_size: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ):
        """One BF16 GEMM task on the shared A/B ring and accumulator stage, by warp role: the AB
        producer warp waits for the operands and loads the K tiles, the UMMA warp of the leader CTA
        issues the MMAs, the epilogue warps store D and thread 0 arrives on the task's readiness
        counters after the stores complete.  ``second``: a second operand pair continues the same
        accumulator (dgrad Gate + Up).  ``d_add_atom`` / ``d_add_tensor``: a Wgrad's reduce-add
        destination, used for every contribution but a ``first_contribution``."""
        if cutlass.const_expr(self.has_tail):
            intermediate_blocks = _ceil_div_i32(intermediate_size, Int32(256))
        else:
            intermediate_blocks = intermediate_size // Int32(256)
        k_tiles = Int32(cute.size(a_tensor, mode=[1]) // 64)
        first_tiles = k_tiles
        if cutlass.const_expr(second):
            k_tiles = k_tiles * Int32(2)
        if cutlass.const_expr(is_wgrad):
            k_tiles = (k_end - k_start) // Int32(64)
        if warp == Int32(AB_TMA_PRODUCER_WARP):
            thr_mma = tiled_mma.get_slice(cta_rank)
            k_offset = Int32(0)
            if cutlass.const_expr(is_wgrad):
                k_offset = k_start
                if kind >= Int32(ROUTED_WGRAD_DOWN):
                    k_offset = k_offset - macro * Int32(macrobatch_size)
            a_base = cute.domain_offset((Int32(0), k_offset), a_tensor)
            b_base = cute.domain_offset((Int32(0), k_offset), b_tensor)
            a_tile = cute.local_tile(a_base, (256, 64), (row, None))
            b_tile = cute.local_tile(b_base, (256, 64), (column, None))
            copy_A = gemm._make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
            copy_B = gemm._make_tma_block_load_fn(b_atom, thr_mma.partition_B(b_tile), sB)
            if cutlass.const_expr(second):
                a2_tile = cute.local_tile(a2_tensor, (256, 64), (row, None))
                b2_tile = cute.local_tile(b2_tensor, (256, 64), (column, None))
                copy_A2 = gemm._make_tma_block_load_fn(a2_atom, thr_mma.partition_A(a2_tile), sA)
                copy_B2 = gemm._make_tma_block_load_fn(b2_atom, thr_mma.partition_B(b2_tile), sB)
            if cutlass.const_expr(not is_wgrad):
                with cute.arch.elect_one():
                    self._wait_task_operands(
                        kind,
                        macro,
                        generation,
                        row,
                        num_tokens,
                        d_y_ready,
                        d_gate_up_ready,
                        replayed_x_ready,
                        num_local_tokens=num_local_tokens,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        macrobatch_size=macrobatch_size,
                        minibatch_size=minibatch_size,
                    )
                cute.arch.sync_warp()
            # QuACK load_tma form with the Wgrad boundary-wait hook per K64 tile.
            peek_empty_status = Boolean(True)
            if k_tiles > Int32(0):
                peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            # Wgrad waits only when a K64 load first touches a producing M256 row
            # block or a minibatch. Track the next boundary tile with adds/compares;
            # no per-tile runtime division.
            row_block_tiles = Int32(256 // 64)
            minibatch_tiles = minibatch_size // Int32(64)
            origin_tile = k_start // Int32(64)
            next_row_block_tile = row_block_tiles - origin_tile % row_block_tiles
            next_minibatch_tile = minibatch_tiles - origin_tile % minibatch_tiles
            operand_row = k_start
            for k in cutlass.range(k_tiles, unroll=1):
                if cutlass.const_expr(is_wgrad):
                    wait_row_block = Int32(0)
                    wait_minibatch = Int32(0)
                    if k == Int32(0):
                        wait_row_block = Int32(1)
                        wait_minibatch = Int32(1)
                    if k == next_row_block_tile:
                        wait_row_block = Int32(1)
                        next_row_block_tile = next_row_block_tile + row_block_tiles
                    if k == next_minibatch_tile:
                        wait_minibatch = Int32(1)
                        next_minibatch_tile = next_minibatch_tile + minibatch_tiles
                    if wait_row_block + wait_minibatch != Int32(0):
                        with cute.arch.elect_one():
                            self._wait_wgrad_operands(
                                kind,
                                macro,
                                wait_row_block,
                                wait_minibatch,
                                operand_row,
                                num_tokens,
                                d_y_ready,
                                d_gate_up_ready,
                                replayed_x_ready,
                                num_local_tokens=num_local_tokens,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                                minibatch_size=minibatch_size,
                            )
                        cute.arch.sync_warp()
                    operand_row = operand_row + Int32(64)
                ab_pipeline.producer_acquire(ab_producer_state, peek_empty_status)
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                stage = ab_producer_state.index
                if cutlass.const_expr(second):
                    if k < first_tiles:
                        copy_A(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr)
                        copy_B(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr)
                    else:
                        copy_A2(src_idx=k - first_tiles, dst_idx=stage, tma_bar_ptr=tma_bar_ptr)
                        copy_B2(src_idx=k - first_tiles, dst_idx=stage, tma_bar_ptr=tma_bar_ptr)
                else:
                    copy_A(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr)
                    copy_B(src_idx=k, dst_idx=stage, tma_bar_ptr=tma_bar_ptr)
                ab_pipeline.producer_commit(ab_producer_state)
                ab_producer_state.advance()
                peek_empty_status = Boolean(True)
                if k + Int32(1) < k_tiles:
                    peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        elif warp == Int32(UMMA_PRODUCER_WARP) and cta_rank == Int32(0):
            fragment_A = tiled_mma.make_fragment_A(sA)
            fragment_B = tiled_mma.make_fragment_B(sB)
            fake = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(collective.mma_tiler[:2]), 1))
            accumulator = cute.make_tensor(acc_ptr, fake.layout)[None, None, None, 0]
            # QuACK MMA mainloop (peek full, MMA, tcgen05.commit release, acc commit).
            ab_consumer_state, acc_producer_state, mma_after = self.abt.mma(
                ab_pipeline,
                acc_pipeline,
                ab_consumer_state,
                acc_producer_state,
                tiled_mma,
                fragment_A,
                fragment_B,
                accumulator,
                k_tiles,
                cta_rank == Int32(0),
                cta_rank,
            )
        elif warp <= Int32(EPILOGUE_WARP_LAST):
            fake = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(collective.mma_tiler[:2]), 1))
            acc_tensor = cute.make_tensor(acc_ptr, fake.layout)
            t2r, acc_partition, r_acc = gemm._Bf16Gemm.epilog_tmem_copy_and_partition(collective, thread, acc_tensor, epi_tile, True)
            r_d = cute.make_rmem_tensor(r_acc.shape, Float32)
            # The destination's dtype comes from the collective: a Wgrad accumulating into a
            # float32 main_grad writes the FP32 accumulator straight out, every other task
            # rounds to bfloat16.
            d_dtype = collective.d_dtype_expected
            r2s, r_d_partition, s_d_partition = gemm._Bf16Gemm.epilog_smem_store_and_partition(collective, t2r, collective.d_layout, d_dtype, r_d, sD, thread)
            acc_partition = cute.group_modes(acc_partition[None, None, None, None, None, 0], 3, 5)
            acc_pipeline.consumer_wait(acc_consumer_state)
            if cutlass.const_expr(d_dtype is Float32):
                # Accumulating into a float32 main_grad: every contribution is a reduce-add, and
                # the accumulator goes out a subtile at a time rather than all eight at once.
                assert d_add_atom is not None, "a float32 D is a main_grad, always accumulated"
                copy_D_add, _, _ = gemm._Bf16Gemm.epilog_gmem_copy_and_partition(
                    collective,
                    d_add_atom,
                    d_add_tensor,
                    collective.cta_tile_shape_mnk[:2],
                    epi_tile,
                    sD,
                    (row * Int32(2) + cta_rank, column, Int32(0), Int32(0)),
                )
                gemm.store_acc_fp32(
                    collective, t2r, r2s, acc_partition, r_acc, r_d_partition, s_d_partition, copy_D_add, thread, acc_pipeline, acc_consumer_state
                )
                acc_consumer_state.advance()
            else:
                # QuACK accumulator hand-off: wait full, load eight N32 fragments,
                # release TMEM on the last one (per-warp elect, both CTAs), advance.
                registers = cute.make_rmem_tensor(cute.append(r_d_partition.shape, 8), d_dtype)
                for subtile in cutlass.range_constexpr(8):
                    self.abt.epi_load_acc_subtile(t2r, r2s, acc_partition, r_acc, r_d_partition, (0, subtile), acc_pipeline, acc_consumer_state, 7)
                    registers[None, None, None, subtile].store(r_d_partition.load().to(d_dtype))
                acc_consumer_state.advance()
                copy_D, _, _ = gemm._Bf16Gemm.epilog_gmem_copy_and_partition(
                    collective, d_atom, d_tensor, collective.cta_tile_shape_mnk[:2], epi_tile, sD, (row * Int32(2) + cta_rank, column, Int32(0), Int32(0))
                )
                if cutlass.const_expr(d_add_atom is not None):
                    copy_D_add, _, _ = gemm._Bf16Gemm.epilog_gmem_copy_and_partition(
                        collective,
                        d_add_atom,
                        d_add_tensor,
                        collective.cta_tile_shape_mnk[:2],
                        epi_tile,
                        sD,
                        (row * Int32(2) + cta_rank, column, Int32(0), Int32(0)),
                    )
                    gemm.store_acc_all(collective, registers, r2s, s_d_partition, copy_D, thread, copy_D_add=copy_D_add, first_contribution=first_contribution)
                else:
                    gemm.store_acc_all(collective, registers, r2s, s_d_partition, copy_D, thread)
            publishes = (
                kind == Int32(SHARED_DOWN_DGRAD)
                or kind == Int32(ROUTED_DOWN_DGRAD)
                or kind == Int32(ROUTED_GATE_UP_DGRAD)
                or kind == Int32(REPLAY_GATE)
                or kind == Int32(REPLAY_UP)
            )
            retires = kind >= Int32(ROUTED_DOWN_DGRAD) and num_tokens > Int32(macrobatch_size)
            if publishes or retires:
                if cutlass.const_expr(d_add_atom is not None):
                    if warp == Int32(0):
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                        cute.arch.sync_warp()
                else:
                    gemm.finish_store(thread)
            if thread == Int32(0):
                global_row = macro * Int32(macrobatch_size // 256) + row
                if kind == Int32(SHARED_DOWN_DGRAD):
                    _counter_arrive_gpu(d_hidden_ready, row * intermediate_blocks + column)
                elif kind == Int32(ROUTED_DOWN_DGRAD):
                    _counter_arrive_gpu(d_hidden_ready, Int32(num_local_tokens // 256 * intermediate_blocks) + global_row * intermediate_blocks + column)
                elif kind == Int32(ROUTED_GATE_UP_DGRAD):
                    _counter_arrive_gpu(d_x_ready, generation)
                elif kind == Int32(REPLAY_GATE) or kind == Int32(REPLAY_UP):
                    _counter_arrive_gpu(replayed_gate_up_ready, global_row * intermediate_blocks + column)
                if retires:
                    _counter_arrive_gpu(routed_buffers_done, macro)
        return ab_producer_state, ab_consumer_state, acc_producer_state, acc_consumer_state


@cute.kernel
def _zero_empty_routed_wgrads(
    gate: cute.Tensor,
    up: cute.Tensor,
    down: cute.Tensor,
    counts: cute.Tensor,
    hidden_size: Int32,
    intermediate_size: Int32,
    num_local_experts: Int32,
    fc1_rows: Int32,
):
    """Zeroes the routed weight gradients of experts without rows (grid y = expert, grid-stride
    over the elements along x).

    ``fc1_rows``: the rows per expert of the Gate and Up buffers -- ``intermediate_size``, or twice
    that when they are one combined FC1 buffer passed under both names (zeroed twice, harmlessly).
    Down's are always ``intermediate_size`` columns of ``hidden_size`` rows.
    """
    expert = cute.arch.block_idx()[1]
    if counts[expert] == Int32(0):
        fc1_elements = Int64(hidden_size) * Int64(fc1_rows)
        down_elements = Int64(hidden_size) * Int64(intermediate_size)
        flat_gate = cute.make_tensor(gate.iterator, cute.make_layout(Int64(num_local_experts) * fc1_elements))
        flat_up = cute.make_tensor(up.iterator, cute.make_layout(Int64(num_local_experts) * fc1_elements))
        flat_down = cute.make_tensor(down.iterator, cute.make_layout(Int64(num_local_experts) * down_elements))
        # CuTe 4.5.2 range lowering requires Int32; keep wide grid-stride math.
        idx = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        element = idx
        while element < fc1_elements:
            offset = Int64(expert) * fc1_elements + Int64(element)
            flat_gate[offset] = BFloat16(0)
            flat_up[offset] = BFloat16(0)
            element += Int64(128 * 256)
        element = idx
        while element < down_elements:
            flat_down[Int64(expert) * down_elements + Int64(element)] = BFloat16(0)
            element += Int64(128 * 256)
