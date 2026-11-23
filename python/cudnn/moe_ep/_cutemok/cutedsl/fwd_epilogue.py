# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""BF16 forward epilogue: output = shared output (optionally gated) + the top-k weighted routed rows.

Persistent 128-thread CTAs stream (token, 1024-column) tiles through a ring of
shared-memory slots (two per CTA, as many CTAs per SM as fit): a slot holds one
tile's topk+1 input vectors, topk weights and topk route ids; the reduced tile
is stored from the slot's first vector, the slot's routed vectors are refilled
with the tile num_slots ahead at once and its first vector one tile later
(after the store has read it), so one tile loads while the other is reduced.
Each thread sums 8 adjacent columns in FP32 in route order and rounds once; an
unused route slot (top_experts < 0) is neither loaded nor summed. Runtime
T/H/topk, tile count and slot count drive the grid, shared-memory size, loops
and addresses. Rows move as 1D bulk copies (one per 2 KB vector, one issuing
lane per vector); an H tail tile moves only its valid bytes.
"""

import threading


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device

TILE_COLUMNS = 1024
THREADS_PER_CTA = 128
COLUMNS_PER_THREAD = TILE_COLUMNS // THREADS_PER_CTA
BF16_BYTES = 2
TILE_BYTES = TILE_COLUMNS * BF16_BYTES
MBARRIER_AREA_BYTES = 1024
CLUSTER_SHAPE = (1, 1, 1)
# Two ring slots per CTA (one tile loads while the other is reduced and stored) and as many resident CTAs
# per SM as fit them, at most six: the multi-CTA bands divide the SM's shared memory (1024 B reserved per
# block), the single-CTA band uses the per-block opt-in limit. With 233472 B per SM and a 232448 B opt-in
# limit: top-k 1-7 run 6 CTAs per SM, 8-9 5, 10-12 4, 13-17 3, 18-26 2, 27-55 1.
MAX_RESIDENT_CTAS_PER_SM = 6
RING_SLOTS = 2
DRIVER_RESERVED_SMEM_BYTES = 1024
# Thread 0 issues the stores, waits for them and then loads the deferred first vector of a refill; warp 1's
# lanes issue the other loads.
LOAD_WARP = 1
# A tile's topk <= 255 weights are prefetched into registers one tile ahead: two per thread cover topk <= 256.
# The route ids of the same tile ride along in a second pair, so a slot carries both when it is reduced.
WEIGHT_REGISTERS = 2

_CODEGEN_EPOCH = "fwd-epilogue-bf16-runtime-v17-persistent-1d-o2-routed-loads"


def _slot_bytes(topk: int) -> int:
    # One tile: topk+1 aligned 2048 B vectors (vectors of all slots first, then the FP32 weights of all slots,
    # then their INT32 route ids).  Dynamic shared memory = MBARRIER_AREA_BYTES + num_slots * _slot_bytes(topk);
    # no static model-sized allocation.
    return (topk + 1) * TILE_BYTES + topk * 4 + topk * 4


def _launch_geometry(device_properties, topk: int) -> tuple[int, int]:
    """Return ``(num_slots, ctas_per_sm)``: RING_SLOTS slots and the most resident CTAs per SM whose shares of
    the SM's shared memory hold them, down to one CTA (the per-block opt-in limit).

    The kernel needs at least two slots: a refill's first vector is loaded one tile after its routed vectors,
    into the slot of the tile before the current one, and with one slot that wait would never return.
    Correctness does not depend on the CTAs being co-resident (there is no grid-wide synchronization; every CTA
    only walks its own tiles), so a device that fits fewer CTAs per SM than this formula assumes would only run
    the grid in a second wave."""

    assert RING_SLOTS >= 2
    ring = MBARRIER_AREA_BYTES + RING_SLOTS * _slot_bytes(topk)
    for ctas_per_sm in range(MAX_RESIDENT_CTAS_PER_SM, 1, -1):
        if ring <= device_properties.shared_memory_per_multiprocessor // ctas_per_sm - DRIVER_RESERVED_SMEM_BYTES:
            return RING_SLOTS, ctas_per_sm
    if ring <= device_properties.shared_memory_per_block_optin:
        return RING_SLOTS, 1
    raise ValueError("topk requires more dynamic shared memory than the device supports")


from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import from_dlpack
from ._host_common import persistent_executor
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import ValidatedCalls, current_stream_handle, tensor_layout_signature
from ._host_common import current_device_index as _current_device_index, device_properties as _device_properties
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw


@cute.jit
def _tile_bytes(hidden_size: Int32, column_block: Int32) -> Int32:
    """Bytes of one row of a tile: 2048, or the H tail (H % 128 == 0 keeps it a multiple of 256)."""

    return cutlass.min(Int32(TILE_BYTES), (hidden_size - column_block * Int32(TILE_COLUMNS)) * Int32(BF16_BYTES))


@cute.jit
def _routed_slots(top_experts: cute.Tensor, token_index: Int32, topk: Int32) -> Int32:
    """How many of the token's route slots are used (top_experts >= 0): the routed vectors a tile loads."""
    count = Int32(0)
    for route_index in cutlass.range(topk):
        if top_experts[Int64(token_index), Int64(route_index)] >= Int32(0):
            count = count + Int32(1)
    return count


@cute.jit
def _issue_tile_loads(
    y_base: Int64,
    combine_base: Int64,
    row_bytes: Int64,
    token_index: Int32,
    column_block: Int32,
    tile_bytes: Int32,
    topk: Int32,
    slot_vectors: cute.Pointer,
    input_arrived: cute.Pointer,
    lane: Int32,
    first_vector: Int32,
    top_experts: cute.Tensor,
) -> None:
    """One lane per vector (several passes when topk+1 > 32), from vector ``first_vector``: vector 0 is the
    shared output row, vector 1+k the routed row token*topk + k -- loaded only for a used slot (k with
    top_experts >= 0); the slot's barrier expects exactly those bytes (_routed_slots)."""

    column_offset = Int64(column_block) * Int64(TILE_BYTES)
    for vector_index in cutlass.range(lane + first_vector, topk + Int32(1), 32):
        destination = slot_vectors + Int64(vector_index) * Int64(TILE_BYTES)
        if vector_index == Int32(0):
            tma_load_1d_raw(destination, y_base + Int64(token_index) * row_bytes + column_offset, input_arrived, tile_bytes)
        else:
            if top_experts[Int64(token_index), Int64(vector_index - Int32(1))] >= Int32(0):
                routed_row = Int64(token_index) * Int64(topk) + Int64(vector_index - Int32(1))
                tma_load_1d_raw(destination, combine_base + routed_row * row_bytes + column_offset, input_arrived, tile_bytes)


@cute.jit
def _slot_weights(slot_weights: cute.Pointer, topk: Int32) -> cute.Tensor:
    return cute.make_tensor(
        cute.recast_ptr(slot_weights, dtype=Float32),
        cute.make_layout((topk,), stride=(1,)),
    )


@cute.jit
def _slot_routes(slot_routes: cute.Pointer, topk: Int32) -> cute.Tensor:
    return cute.make_tensor(
        cute.recast_ptr(slot_routes, dtype=Int32),
        cute.make_layout((topk,), stride=(1,)),
    )


@cute.jit
def _load_tile_routing(
    topk_weights: cute.Tensor,
    top_experts: cute.Tensor,
    slot_weights: cute.Pointer,
    slot_routes: cute.Pointer,
    token_index: Int32,
    topk: Int32,
    thread: Int32,
) -> None:
    """Prologue only: the weights and route ids of one tile straight from global memory into its slot."""

    weights = _slot_weights(slot_weights, topk)
    routes = _slot_routes(slot_routes, topk)
    for route_index in cutlass.range(thread, topk, THREADS_PER_CTA):
        weights[route_index] = topk_weights[Int64(token_index), Int64(route_index)]
        routes[route_index] = top_experts[Int64(token_index), Int64(route_index)]


@cute.jit
def _prefetch_tile_routing(
    topk_weights: cute.Tensor,
    top_experts: cute.Tensor,
    token_index: Int32,
    topk: Int32,
    thread: Int32,
    registers: cute.Tensor,
    route_registers: cute.Tensor,
) -> None:
    """Issue this thread's global loads of a tile's weights and route ids (routes thread, thread+128); they land in
    registers while the current tile is processed, off the per-tile critical path."""

    for index in cutlass.range_constexpr(WEIGHT_REGISTERS):
        route_index = thread + Int32(index * THREADS_PER_CTA)
        if route_index < topk:
            registers[index] = topk_weights[Int64(token_index), Int64(route_index)]
            route_registers[index] = top_experts[Int64(token_index), Int64(route_index)]


@cute.jit
def _store_tile_routing(
    slot_weights: cute.Pointer,
    slot_routes: cute.Pointer,
    topk: Int32,
    thread: Int32,
    registers: cute.Tensor,
    route_registers: cute.Tensor,
) -> None:
    weights = _slot_weights(slot_weights, topk)
    routes = _slot_routes(slot_routes, topk)
    for index in cutlass.range_constexpr(WEIGHT_REGISTERS):
        route_index = thread + Int32(index * THREADS_PER_CTA)
        if route_index < topk:
            weights[route_index] = registers[index]
            routes[route_index] = route_registers[index]


class _FwdEpilogue:
    """The epilogue kernel and its two compile-time variant flags.

    ``gated``: the shared output is scaled by its per-token output gate before the routed sum: the
    saved BF16 shared output and gate are multiplied in FP32 and the accumulator stays FP32 until
    the BF16 store.  The ungated variant's kernel takes no gate (the launch passes a placeholder).

    ``has_shared=False``: a routed-only layer: the accumulator starts at zero and only the routed
    rows are loaded.  Slot vector 0 still stages the output tile; the shared argument is a
    placeholder never read."""

    def __init__(self, gated: bool = False, has_shared: bool = True) -> None:
        if gated and not has_shared:
            raise ValueError("the shared output gate requires a shared output")
        self.gated = bool(gated)
        self.has_shared = bool(has_shared)

    @cute.jit
    def __call__(
        self,
        y_shared: cute.Tensor,
        combine_buffer: cute.Tensor,
        topk_weights: cute.Tensor,
        top_experts: cute.Tensor,
        shared_output_gate: cute.Tensor,
        output: cute.Tensor,
        hidden_size: Int32,
        topk: Int32,
        num_tiles: Int32,
        num_slots: Int32,
        grid_size: Int32,
        stream: cuda.CUstream,
    ) -> None:
        # Only the gated variant's kernel takes the gate: None is a compile-time constant, not a kernel parameter.
        kernel_gate = None
        if cutlass.const_expr(self.gated):
            kernel_gate = shared_output_gate
        self.kernel(
            y_shared,
            combine_buffer,
            topk_weights,
            top_experts,
            kernel_gate,
            output,
            hidden_size,
            topk,
            num_tiles,
            num_slots,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=MBARRIER_AREA_BYTES + num_slots * ((topk + 1) * TILE_BYTES + topk * 4 + topk * 4),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        y_shared: cute.Tensor,
        combine_buffer: cute.Tensor,
        topk_weights: cute.Tensor,
        top_experts: cute.Tensor,
        shared_output_gate,  # gated variant only, else None (hence no annotation)
        output: cute.Tensor,
        hidden_size: Int32,
        topk: Int32,
        num_tiles: Int32,
        num_slots: Int32,
    ) -> None:
        thread = cute.arch.thread_idx()[0]
        lane = Int32(cute.arch.lane_idx())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block = cute.arch.make_warp_uniform(cute.arch.block_idx()[0])
        grid = cute.arch.make_warp_uniform(cute.arch.grid_dim()[0])
        column_blocks = (hidden_size + Int32(TILE_COLUMNS - 1)) // Int32(TILE_COLUMNS)
        num_input_vectors = topk + Int32(1)
        # The vectors a tile's loads deliver: the routed rows, and the shared row when there is one.
        shared_vectors = Int32(int(self.has_shared))  # + the tile's used route slots (_routed_slots)
        first_loaded_vector = Int32(0) if cutlass.const_expr(self.has_shared) else Int32(1)
        # The host launches grid <= num_tiles CTAs: this CTA walks tiles block, block+grid, ... (at least one);
        # a tile's (token, column block) advances by (grid_step_tokens, grid_step_columns) with a carry.
        my_tiles = (num_tiles - block + grid - Int32(1)) // grid
        prologue_tiles = cutlass.min(num_slots, my_tiles)
        grid_step_tokens = grid // column_blocks
        grid_step_columns = grid - grid_step_tokens * column_blocks
        first_token = block // column_blocks
        first_column = block - first_token * column_blocks

        y_base = y_shared.iterator.toint()
        combine_base = combine_buffer.iterator.toint()
        output_base = output.iterator.toint()
        row_bytes = Int64(hidden_size) * Int64(BF16_BYTES)

        # Reserve the first 1024B for the slot mbarriers. The vectors of all slots come first (each 2048B
        # vector start stays 16B-aligned for the bulk copies), then the FP32 weights of all slots, then
        # their INT32 route ids.
        base = cute.arch.get_dyn_smem(Uint8, alignment=MBARRIER_AREA_BYTES)
        input_arrived = cute.recast_ptr(base, dtype=Int64)
        arena = base + Int64(MBARRIER_AREA_BYTES)
        slot_vector_bytes = Int64(num_input_vectors) * Int64(TILE_BYTES)
        weights_arena = arena + Int64(num_slots) * slot_vector_bytes
        slot_weight_bytes = Int64(topk) * Int64(4)
        routes_arena = weights_arena + Int64(num_slots) * slot_weight_bytes
        slot_route_bytes = slot_weight_bytes

        # Prologue: arm the first min(num_slots, my_tiles) slots, make the barriers visible, then issue their loads
        # before the weights are read from global memory, so that the weights' latency overlaps the first loads
        # (every CTA of the persistent grid starts at once, so a serial prologue is on the critical path).
        token_index = first_token
        column_block = first_column
        if thread == Int32(0):
            for slot in cutlass.range(num_slots):
                cute.arch.mbarrier_init(input_arrived + slot, 1)
            for slot in cutlass.range(prologue_tiles):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    input_arrived + slot, (shared_vectors + _routed_slots(top_experts, token_index, topk)) * _tile_bytes(hidden_size, column_block)
                )
                token_index = token_index + grid_step_tokens
                column_block = column_block + grid_step_columns
                if column_block >= column_blocks:
                    column_block = column_block - column_blocks
                    token_index = token_index + Int32(1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        token_index = first_token
        column_block = first_column
        if warp == Int32(LOAD_WARP):
            for slot in cutlass.range(prologue_tiles):
                _issue_tile_loads(
                    y_base,
                    combine_base,
                    row_bytes,
                    token_index,
                    column_block,
                    _tile_bytes(hidden_size, column_block),
                    topk,
                    arena + Int64(slot) * slot_vector_bytes,
                    input_arrived + slot,
                    lane,
                    first_loaded_vector,
                    top_experts,
                )
                token_index = token_index + grid_step_tokens
                column_block = column_block + grid_step_columns
                if column_block >= column_blocks:
                    column_block = column_block - column_blocks
                    token_index = token_index + Int32(1)
        token_index = first_token
        column_block = first_column
        for slot in cutlass.range(prologue_tiles):
            _load_tile_routing(
                topk_weights,
                top_experts,
                weights_arena + Int64(slot) * slot_weight_bytes,
                routes_arena + Int64(slot) * slot_route_bytes,
                token_index,
                topk,
                thread,
            )
            token_index = token_index + grid_step_tokens
            column_block = column_block + grid_step_columns
            if column_block >= column_blocks:
                column_block = column_block - column_blocks
                token_index = token_index + Int32(1)
        cute.arch.sync_threads()
        # (token_index, column_block) is now the tile of ordinal prologue_tiles: ordinal num_slots when the ring
        # is full, the first tile the main loop prefetches weights for.
        next_token = token_index
        next_column = column_block
        token_index = first_token
        column_block = first_column

        # (next_token, next_column) is the tile num_slots ahead of the current one: the refill of the slot being
        # reduced. Its weights are prefetched into registers one tile earlier; its routed vectors are loaded
        # right after the barrier, its first vector one tile later, once thread 0 has seen the store read it.
        pending_weights = cute.make_rmem_tensor((WEIGHT_REGISTERS,), Float32)
        pending_routes = cute.make_rmem_tensor((WEIGHT_REGISTERS,), Int32)
        if num_slots < my_tiles:
            _prefetch_tile_routing(topk_weights, top_experts, next_token, topk, thread, pending_weights, pending_routes)
        slot = Int32(0)
        phase = Int32(0)
        pending_first = Int32(0)  # the previous tile's refill still owes its first vector
        pending_slot = Int32(0)
        pending_token = Int32(0)
        pending_column = Int32(0)
        for ordinal in cutlass.range(my_tiles):
            tile_bytes = _tile_bytes(hidden_size, column_block)
            slot_vectors = arena + Int64(slot) * slot_vector_bytes
            token_vectors = cute.make_tensor(
                cute.recast_ptr(slot_vectors, dtype=BFloat16),
                cute.make_layout((num_input_vectors, TILE_COLUMNS), stride=(TILE_COLUMNS, 1)),
            )
            weights = _slot_weights(weights_arena + Int64(slot) * slot_weight_bytes, topk)
            routes = _slot_routes(routes_arena + Int64(slot) * slot_route_bytes, topk)

            cute.arch.mbarrier_wait(input_arrived + slot, phase)
            # Each thread owns COLUMNS_PER_THREAD adjacent columns, so a route's values are one vector
            # shared-memory load; per element the FP32 sum runs in route order (the order fixes the rounded
            # result). Columns past an H tail hold stale shared memory and are never stored.
            first_thread_column = thread * Int32(COLUMNS_PER_THREAD)
            accumulators = cute.make_rmem_tensor((COLUMNS_PER_THREAD,), Float32)
            for element in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                if cutlass.const_expr(self.has_shared):
                    accumulators[element] = Float32(token_vectors[0, first_thread_column + Int32(element)])
                else:
                    # Routed-only: vector 0 is never loaded, only overwritten with the output below.
                    # The store that last read it is the slot's previous tile's, num_slots >= 2
                    # tiles back, whose read thread 0 awaited in the previous iteration's
                    # wait_group, ordered before this one by that iteration's barrier.
                    accumulators[element] = Float32(0.0)
            if cutlass.const_expr(self.gated):
                # Multiply the saved BF16 shared output and gate in FP32; keep the accumulator
                # FP32 until the final BF16 store.
                gate = Float32(shared_output_gate[Int64(token_index), Int64(0)])
                for element in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                    accumulators[element] = accumulators[element] * gate
            for route_index in cutlass.range(topk):
                # A token may route to fewer than topk experts; the schedule marks the empty slots -1 and
                # the dispatch never writes their combine rows, so those hold whatever the symmetric buffer
                # held before (it is allocated, not zeroed).  Weighting by zero would not be enough: a stale
                # NaN times zero is still NaN.
                if routes[route_index] >= Int32(0):
                    weight = Float32(weights[route_index])
                    for element in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                        term = Float32(token_vectors[route_index + 1, first_thread_column + Int32(element)])
                        accumulators[element] = accumulators[element] + term * weight
            for element in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                token_vectors[0, first_thread_column + Int32(element)] = accumulators[element].to(BFloat16)
            cute.arch.fence_view_async_shared()

            # Thread 0: the previous tile's store has read its slot's first vector once this wait returns (this
            # tile's store is issued after the barrier below), so the first vector of that slot's refill can
            # be loaded now; then the slot being reduced is re-armed for the tile num_slots ahead. Its phase
            # has completed (thread 0 is past its own wait), so a thread still in its wait returns on the old
            # parity; the refill's loads come after the barrier.
            refill = ordinal + num_slots < my_tiles
            refill_bytes = _tile_bytes(hidden_size, next_column)
            # Routed-only, a refill tile whose token has no used route slot expects no bytes. Arming it here
            # would complete the new phase at once and bring the barrier back to the parity the other warps
            # are still testing in this tile's wait, which would then never return. Thread 0 arrives for such
            # a tile after the barrier below instead, when every thread is past its wait.
            empty_refill = Int32(0)
            if thread == Int32(0):
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                if pending_first == Int32(1):
                    tma_load_1d_raw(
                        arena + Int64(pending_slot) * slot_vector_bytes,
                        y_base + Int64(pending_token) * row_bytes + Int64(pending_column) * Int64(TILE_BYTES),
                        input_arrived + pending_slot,
                        _tile_bytes(hidden_size, pending_column),
                    )
                if refill:
                    refill_vectors = shared_vectors + _routed_slots(top_experts, next_token, topk)
                    if cutlass.const_expr(self.has_shared):
                        # At least the shared row: the phase stays open until the loads issued after the barrier.
                        cute.arch.mbarrier_arrive_and_expect_tx(input_arrived + slot, refill_vectors * refill_bytes)
                    else:
                        if refill_vectors > Int32(0):
                            cute.arch.mbarrier_arrive_and_expect_tx(input_arrived + slot, refill_vectors * refill_bytes)
                        else:
                            empty_refill = Int32(1)
            cute.arch.sync_threads()
            if thread == Int32(0):
                tma_store_1d_raw(
                    output_base + Int64(token_index) * row_bytes + Int64(column_block) * Int64(TILE_BYTES),
                    slot_vectors,
                    tile_bytes,
                )
                cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(not self.has_shared):
                    if empty_refill == Int32(1):
                        # No load follows for this tile: the arrive alone completes the slot's next phase.
                        cute.arch.mbarrier_arrive(input_arrived + slot)
            # The refill's weights (readers are at least two tiles away, so one barrier apart for any ring
            # depth) and routed vectors go into the slot just reduced; its first vector follows next tile.
            if refill:
                _store_tile_routing(
                    weights_arena + Int64(slot) * slot_weight_bytes,
                    routes_arena + Int64(slot) * slot_route_bytes,
                    topk,
                    thread,
                    pending_weights,
                    pending_routes,
                )
                if warp == Int32(LOAD_WARP):
                    _issue_tile_loads(
                        y_base,
                        combine_base,
                        row_bytes,
                        next_token,
                        next_column,
                        refill_bytes,
                        topk,
                        slot_vectors,
                        input_arrived + slot,
                        lane,
                        Int32(1),
                        top_experts,
                    )
            pending_first = Int32(0)
            if cutlass.const_expr(self.has_shared):
                if refill:
                    pending_first = Int32(1)
            pending_slot = slot
            pending_token = next_token
            pending_column = next_column

            # Advance: the tile num_slots ahead (and prefetch its weights), the ring slot and its phase, the
            # current tile.
            next_token = next_token + grid_step_tokens
            next_column = next_column + grid_step_columns
            if next_column >= column_blocks:
                next_column = next_column - column_blocks
                next_token = next_token + Int32(1)
            if ordinal + num_slots + Int32(1) < my_tiles:
                _prefetch_tile_routing(topk_weights, top_experts, next_token, topk, thread, pending_weights, pending_routes)
            slot = slot + Int32(1)
            if slot == num_slots:
                slot = Int32(0)
                phase = phase ^ Int32(1)
            token_index = token_index + grid_step_tokens
            column_block = column_block + grid_step_columns
            if column_block >= column_blocks:
                column_block = column_block - column_blocks
                token_index = token_index + Int32(1)

        # Shared memory may be released once the last store has read it; the global writes complete with
        # the kernel.
        if thread == Int32(0):
            cute.arch.cp_async_bulk_wait_group(0, read=True)


# (gated, has_shared) -> the compile-time variant.
_EPILOGUES = {
    (False, True): _FwdEpilogue(gated=False),
    (True, True): _FwdEpilogue(gated=True),
    (False, False): _FwdEpilogue(gated=False, has_shared=False),
}
_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


def _validate_inputs(
    y_shared,
    combine_buffer,
    topk_weights,
    top_experts,
) -> tuple[object, int, int, int, int, int, int, int]:
    import torch

    for name, tensor in (
        ("combine_buffer", combine_buffer),
        ("topk_weights", topk_weights),
        ("top_experts", top_experts),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if y_shared is not None and not isinstance(y_shared, torch.Tensor):
        raise TypeError("y_shared must be a torch.Tensor or None")
    if not combine_buffer.is_cuda:
        raise ValueError("combine_buffer must be a CUDA tensor")
    device = combine_buffer.device
    if combine_buffer.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("combine_buffer must have shape (T * topk, H) and topk_weights (T, topk)")
    # From the routed side: a routed-only layer has no y_shared.
    num_tokens, hidden_size = int(topk_weights.shape[0]), int(combine_buffer.shape[1])
    if y_shared is not None:
        if not y_shared.is_cuda or y_shared.device != device:
            raise ValueError("all inputs must be on the same CUDA device")
        if y_shared.dtype != torch.bfloat16:
            raise NotImplementedError("CuTe DSL forward epilogue requires BF16 activations")
        if tuple(y_shared.shape) != (num_tokens, hidden_size):
            raise ValueError(f"y_shared must have shape (T, H) = {(num_tokens, hidden_size)}; got {list(y_shared.shape)}")
        if not y_shared.is_contiguous():
            raise ValueError("y_shared must be contiguous")
    if hidden_size <= 0 or hidden_size % 128 != 0:
        raise ValueError(f"hidden_size must be positive and divisible by 128; got {hidden_size}")
    if num_tokens < 512 or num_tokens % 256 != 0:
        raise NotImplementedError("CuTe DSL forward epilogue requires T>=512 with T divisible by 256")
    if combine_buffer.device != device or topk_weights.device != device or top_experts.device != device:
        raise ValueError("all inputs must be on the same CUDA device")
    if combine_buffer.dtype != torch.bfloat16:
        raise NotImplementedError("CuTe DSL forward epilogue requires BF16 activations")
    topk = int(topk_weights.shape[1])
    if not 0 < topk <= 255:
        raise ValueError(f"topk must be in [1, 255]; got {topk}")
    if tuple(combine_buffer.shape) != (num_tokens * topk, hidden_size):
        raise ValueError("combine_buffer must have shape (T * topk, H); got " f"{list(combine_buffer.shape)} for T={num_tokens}, topk={topk}, H={hidden_size}")
    column_blocks = (hidden_size + TILE_COLUMNS - 1) // TILE_COLUMNS
    num_tiles = num_tokens * column_blocks
    # Row indices, the tile count and the grid are signed32, unlike byte strides/offsets.
    if num_tokens * topk >= 2**31 or hidden_size + TILE_COLUMNS - 1 >= 2**31:
        raise ValueError("epilogue dimensions exceed signed32 row coordinates")
    if num_tiles >= 2**31:
        raise ValueError("epilogue tile count exceeds signed32")
    if not combine_buffer.is_contiguous():
        raise ValueError("combine_buffer must be contiguous")
    if top_experts.shape != topk_weights.shape:
        raise ValueError(f"top_experts must have topk_weights' shape {list(topk_weights.shape)}; " f"got {list(top_experts.shape)}")
    if top_experts.dtype != torch.int32:
        raise ValueError("top_experts must be int32")
    if not top_experts.is_contiguous():
        raise ValueError("top_experts must be contiguous")
    if topk_weights.dtype != torch.float32:
        raise NotImplementedError("CuTe DSL forward epilogue requires FP32 weights")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    if any(tensor.data_ptr() % 16 for tensor in (y_shared, combine_buffer, topk_weights, top_experts) if tensor is not None):
        raise ValueError("CuTe DSL forward epilogue inputs must be 16-byte aligned")
    device_index = device.index if device.index is not None else _current_device_index()
    if _current_device_index() != device_index:
        raise RuntimeError("the input CUDA device must be current")
    properties = _device_properties(device)
    num_slots, ctas_per_sm = _launch_geometry(properties, topk)
    grid_size = min(num_tiles, properties.multi_processor_count * ctas_per_sm)
    return device, device_index, num_tokens, hidden_size, topk, num_tiles, num_slots, grid_size


def _make_arguments(
    y_shared,
    combine_buffer,
    topk_weights,
    top_experts,
    shared_output_gate,
    output,
    hidden_size,
    topk,
    num_tiles,
    num_slots,
    grid_size,
    stream,
):
    # A contiguous [T,1] view may have an arbitrary singleton-axis stride.
    # as_strided changes metadata only, retaining storage and storage_offset.
    topk_weights = topk_weights.as_strided(topk_weights.shape, (topk, 1))
    top_experts = top_experts.as_strided(top_experts.shape, (topk, 1))
    # Fixed dtype/rank/unit stride; all extents and other strides are dynamic.
    return (
        from_dlpack(y_shared, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(combine_buffer, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(topk_weights, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(top_experts, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        # [T, 1]: the token is the dynamic mode, the single column the static unit stride.
        from_dlpack(shared_output_gate, assumed_align=2, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(output, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        Int32(hidden_size),
        Int32(topk),
        Int32(num_tiles),
        Int32(num_slots),
        Int32(grid_size),
        cuda.CUstream(stream.cuda_stream),
    )


_VALIDATED_CALLS = ValidatedCalls()
_GATE_PLACEHOLDERS: dict[object, object] = {}


def _gate_placeholder(device):
    """The ungated variant's gate argument: never read, one per device."""
    import torch

    placeholder = _GATE_PLACEHOLDERS.get(device)
    if placeholder is None:
        placeholder = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
        _GATE_PLACEHOLDERS[device] = placeholder
    return placeholder


def _compile_fwd_epilogue_executor(
    y_shared,
    combine_buffer,
    weights_view,
    routes_view,
    gate,
    output,
    hidden_size,
    topk,
    num_tiles,
    num_slots,
    grid_size,
    cache_key,
    gated,
    has_shared,
):
    import torch

    device = combine_buffer.device
    stream = torch.cuda.current_stream(device)
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(cache_key)
        if executor is None:
            check_device(device, "CuTe DSL forward epilogue")
            arguments = _make_arguments(
                y_shared,
                combine_buffer,
                weights_view,
                routes_view,
                gate,
                output,
                hidden_size,
                topk,
                num_tiles,
                num_slots,
                grid_size,
                stream,
            )
            executor = persistent_executor(
                ("fwd_epilogue", cache_key[0], *cache_key[3:]), lambda: cute.compile(_EPILOGUES[gated, has_shared], *arguments, options=_COMPILE_OPTIONS)
            )
            _EXECUTOR_CACHE[cache_key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device.index)
    return executor


def fwd_epilogue_bf16(y_shared, combine_buffer, topk_weights, top_experts, shared_output_gate=None, out=None):
    """Combine the shared output and the top-k weighted routed rows; returns BF16 ``[T, H]``:
    ``y_shared[t] + sum_k topk_weights[t, k] * combine_buffer[t * topk + k]`` (FP32 sum, one rounding).

    ``y_shared``: contiguous BF16 ``[T, H]``, or None for a routed-only layer (the routed sum alone).
    ``combine_buffer``: contiguous BF16 ``[T * topk, H]``.  ``topk_weights``: FP32 ``[T, topk]``.
    T, H and topk are runtime values (T >= 512 and divisible by 256, H divisible by 128, topk <= 255).

    ``shared_output_gate``: the shared expert's per-token output gate, BF16 ``[T, 1]``, or None.
    Given, the shared output is multiplied by it in FP32 before the routed terms are added; the
    output stays BF16.

    ``top_experts`` (int32 ``[T, topk]``) marks a token's unused route slots with -1.  Their combine
    rows were never written, so they are skipped rather than weighted by zero: the symmetric buffer
    is allocated and not zeroed, and a stale NaN times zero is still NaN.

    ``out``: the caller's output buffer, contiguous BF16 ``[T, H]``, 16-byte aligned, not overlapping an input
    (the execute step of a plan); None allocates the output.

    The full input check and the executor key run only when the call signature (layouts, device,
    context) changes; the call passes the torch tensors and plain scalars to the compiled TVM FFI
    function.
    """

    import torch

    has_shared = y_shared is not None
    inputs = (y_shared, combine_buffer, topk_weights, top_experts) if has_shared else (combine_buffer, topk_weights, top_experts)
    try:
        signature = (
            tensor_layout_signature(inputs),
            has_shared,
            _current_device_index(),
            _current_cuda_context_key(),
        )
        validated = _VALIDATED_CALLS.get(signature)  # hashes the signature
    except (AttributeError, TypeError):  # not tensors / unhashable: the full check reports it
        signature = validated = None
    if validated is None:
        device, device_index, num_tokens, hidden_size, topk, num_tiles, num_slots, grid_size = _validate_inputs(
            y_shared,
            combine_buffer,
            topk_weights,
            top_experts,
        )
        validated = (device, hidden_size, topk, num_tiles, num_slots, grid_size, (_CODEGEN_EPOCH, device_index, _current_cuda_context_key()))
        if signature is not None:
            _VALIDATED_CALLS.put(signature, validated)
    elif any(tensor.data_ptr() % 16 for tensor in inputs):
        # Addresses are not part of the signature: keep the alignment check per call.
        raise ValueError("CuTe DSL forward epilogue inputs must be 16-byte aligned")
    device, hidden_size, topk, num_tiles, num_slots, grid_size, cache_key = validated
    gated = shared_output_gate is not None
    if gated and not has_shared:
        raise ValueError("the shared output gate requires a shared output")
    if gated:
        if (
            not isinstance(shared_output_gate, torch.Tensor)
            or not shared_output_gate.is_cuda
            or shared_output_gate.dtype != torch.bfloat16
            or not shared_output_gate.is_contiguous()
            or shared_output_gate.device != y_shared.device
            or tuple(shared_output_gate.shape) != (y_shared.shape[0], 1)
        ):
            raise ValueError("shared_output_gate must be contiguous CUDA BF16 [T, 1] on y_shared's device")
        gate = shared_output_gate
    else:
        gate = _gate_placeholder(device)
    cache_key = cache_key + (("gated", gated), ("has_shared", has_shared))
    if out is None:
        output = torch.empty(topk_weights.shape[0], hidden_size, dtype=torch.bfloat16, device=device)
    else:
        output = out
        if (
            type(output) is not torch.Tensor
            or output.dtype != torch.bfloat16
            or output.device != device
            or tuple(output.shape) != (topk_weights.shape[0], hidden_size)
            or not output.is_contiguous()
            or output.data_ptr() % 16
        ):
            raise ValueError("out must be a contiguous 16-byte aligned CUDA BF16 [T, H] tensor on the input device")
        if any(output.data_ptr() == tensor.data_ptr() for tensor in inputs):
            raise ValueError("out must not be one of the inputs")
    if not has_shared:
        # The absent shared output's argument: the routed buffer's metadata, never loaded.
        y_shared = combine_buffer
    # A contiguous [T,1] view may carry an arbitrary singleton-axis stride; normalize the
    # metadata only (no copy).
    weights_view = topk_weights if topk_weights.stride() == (topk, 1) else topk_weights.as_strided(topk_weights.shape, (topk, 1))
    routes_view = top_experts if top_experts.stride() == (topk, 1) else top_experts.as_strided(top_experts.shape, (topk, 1))
    executor = _EXECUTOR_CACHE.get(cache_key)
    if executor is None:
        executor = _compile_fwd_epilogue_executor(
            y_shared,
            combine_buffer,
            weights_view,
            routes_view,
            gate,
            output,
            hidden_size,
            topk,
            num_tiles,
            num_slots,
            grid_size,
            cache_key,
            gated,
            has_shared,
        )
    executor(
        y_shared,
        combine_buffer,
        weights_view,
        routes_view,
        gate,
        output,
        hidden_size,
        topk,
        num_tiles,
        num_slots,
        grid_size,
        current_stream_handle(device.index),
    )
    # No record_stream: like PyTorch's own ops, the inputs are used on the current stream and a
    # caller that frees them from another stream synchronizes that stream itself.
    return output


def compile_fwd_epilogue(y_shared, combine_buffer, topk_weights, top_experts, *, gated: bool) -> None:
    """The plan step: compile one variant (shared output present or not, gated or not) without launching.
    The arguments are checked as a call's are and only give the kernel's tensor ranks and dtypes, so workspace
    buffers of a call's shapes stand in for the caller's tensors; ``y_shared`` None is the routed-only variant."""
    has_shared = y_shared is not None
    device, device_index, _, hidden_size, topk, num_tiles, num_slots, grid_size = _validate_inputs(y_shared, combine_buffer, topk_weights, top_experts)
    gated = bool(gated)
    cache_key = (_CODEGEN_EPOCH, device_index, _current_cuda_context_key(), ("gated", gated), ("has_shared", has_shared))
    stand_in = y_shared if has_shared else combine_buffer  # the shared output's and the output's argument
    _compile_fwd_epilogue_executor(
        stand_in,
        combine_buffer,
        topk_weights.as_strided(topk_weights.shape, (topk, 1)),
        top_experts.as_strided(top_experts.shape, (topk, 1)),
        _gate_placeholder(device),
        stand_in,
        hidden_size,
        topk,
        num_tiles,
        num_slots,
        grid_size,
        cache_key,
        gated,
        has_shared,
    )


def clear_fwd_epilogue_executor_cache(*, synchronize: bool = True) -> None:
    """Release compiled executors after their launches are quiescent."""

    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _VALIDATED_CALLS.clear()


__all__ = [
    "compile_fwd_epilogue",
    "fwd_epilogue_bf16",
    "clear_fwd_epilogue_executor_cache",
]
